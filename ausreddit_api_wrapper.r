# r Ausreddit API wrapper

library(httr)
library(jsonlite)
library(dplyr)

# Custom error class for API errors
APIError <- function(status_code, message) {
  structure(list(status_code = status_code, message = message),
            class = "APIError")
}

# Function to handle API errors
handle_api_error <- function(response) {
  status_code <- status_code(response)
  message <- content(response, "text", encoding = "UTF-8")
  
  if (status_code == 422) {
    detail <- fromJSON(message)$detail
    stop(APIError(422, paste("Unprocessable Entity:", detail)))
  } else if (status_code >= 400 && status_code < 500) {
    stop(APIError(status_code, paste("Client Error:", message)))
  } else if (status_code >= 500 && status_code < 600) {
    stop(APIError(status_code, paste("Server Error:", message)))
  } else {
    stop(APIError(status_code, paste("HTTP Error:", message)))
  }
}

# API Wrapper class
APIWrapper <- setRefClass("APIWrapper",
  fields = list(
    base_url = "character",
    jwt_file_path = "character",
    session = "ANY",
    token = "character",
    token_expiry = "POSIXct"
  ),
  
  methods = list(
    initialize = function(base_url, jwt_file_path) {
      .self$base_url <- base_url
      .self$jwt_file_path <- jwt_file_path
      .self$session <- httr::new_handle()
      .self$token <- NULL
      .self$token_expiry <- NULL
    },
    
    load_jwt = function() {
      if (!file.exists(.self$jwt_file_path)) {
        stop("JWT file not found at ", .self$jwt_file_path)
      }
      
      jwt_data <- fromJSON(.self$jwt_file_path)
      .self$token <- jwt_data$token
      expiry_str <- jwt_data$expiry
      if (!is.null(expiry_str)) {
        .self$token_expiry <- as.POSIXct(expiry_str, format = "%Y-%m-%dT%H:%M:%S")
      }
      
      if (is.null(.self$token)) {
        stop("No token found in JWT file")
      }
    },
    
    is_token_valid = function() {
      if (is.null(.self$token) || is.null(.self$token_expiry)) {
        return(FALSE)
      }
      return(Sys.time() < .self$token_expiry)
    },
    
    get_auth_header = function() {
      if (!.self$is_token_valid()) {
        .self$load_jwt()
      }
      return(c(Authorization = paste("Bearer", .self$token)))
    },
    
    make_request = function(endpoint, params = NULL, method = 'GET', data = NULL) {
      url <- paste0(.self$base_url, "/", endpoint)
      headers <- .self$get_auth_header()
      
      response <- NULL
      if (method == 'GET') {
        response <- GET(url, query = params, add_headers(.headers = headers))
      } else if (method == 'POST') {
        response <- POST(url, query = params, add_headers(.headers = headers), body = data, encode = "json")
      } else {
        stop("Unsupported HTTP method: ", method)
      }
      
      if (http_type(response) != "application/json") {
        handle_api_error(response)
      }
      
      if (http_error(response)) {
        handle_api_error(response)
      }
      
      return(content(response, "parsed"))
    },
    
    list_subreddits = function(meta = FALSE) {
      params <- list(meta = meta)
      tryCatch({
        response <- .self$make_request('subreddits', params)
        return(.self$process_subreddit_response(response))
      }, APIError = function(e) {
        cat("Error listing subreddits:", e$message, "\n")
        return(data.frame())  # Return an empty DataFrame on error
      })
    },
    
    process_subreddit_response = function(response) {
      subreddits <- response$subreddits
      processed_data <- lapply(subreddits, function(subreddit) {
        list(
          id = subreddit$id,
          display_name = subreddit$display_name,
          title = subreddit$title,
          description = subreddit$description,
          public_description = subreddit$public_description,
          created_utc = .self$parse_unix_timestamp(subreddit$created_utc),
          subscribers = as.integer(subreddit$subscribers),
          over18 = subreddit$over18,
          url = subreddit$url,
          banner_img = subreddit$banner_img,
          icon_img = subreddit$icon_img,
          community_icon = subreddit$community_icon,
          lang = subreddit$lang
        )
      })
      
      return(as.data.frame(do.call(rbind, processed_data)))
    },
    
    parse_unix_timestamp = function(timestamp) {
      if (!is.null(timestamp)) {
        return(as.POSIXct(as.integer(timestamp), origin = "1970-01-01", tz = "UTC"))
      }
      return(NULL)
    },
    
    get_submissions = function(subreddit_id) {
      tryCatch({
        response <- .self$make_request(paste0('submissions/', subreddit_id))
        return(.self$process_submission_response(response))
      }, APIError = function(e) {
        cat("Error getting submissions:", e$message, "\n")
        return(data.frame())
      })
    },
    
    process_submission_response = function(response) {
      submissions <- response$submissions
      processed_data <- lapply(submissions, function(submission) {
        list(
          id = submission$id,
          title = submission$title,
          selftext = submission$selftext,
          author = submission$author,
          created_utc = .self$parse_unix_timestamp(submission$created_utc),
          retrieved_utc = .self$parse_unix_timestamp(submission$retrieved_utc),
          permalink = submission$permalink,
          url = submission$url,
          score = as.integer(submission$score),
          over_18 = submission$over_18,
          subreddit_id = submission$subreddit_id,
          subreddit = submission$subreddit,
          comment_count = as.integer(submission$comment_count)
        )
      })
      
      return(as.data.frame(do.call(rbind, processed_data)))
    },
    
    get_comments = function(submission_id) {
      tryCatch({
        response <- .self$make_request(paste0('comments/', submission_id))
        return(.self$process_comment_response(response))
      }, APIError = function(e) {
        cat("Error getting comments:", e$message, "\n")
        return(data.frame())
      })
    },
    
    process_comment_response = function(response) {
      comments <- response$comments
      processed_data <- lapply(comments, function(comment) {
        list(
          id = comment$id,
          author = comment$author,
          body = comment$body,
          created_utc = .self$parse_unix_timestamp(comment$created_utc),
          link_id = comment$link_id,
          parent_id = comment$parent_id,
          score = as.integer(comment$score),
          subreddit_id = comment$subreddit_id,
          subreddit = comment$subreddit,
          permalink = comment$permalink,
          retrieved_utc = .self$parse_unix_timestamp(comment$retrieved_utc)
        )
      })
      
      return(as.data.frame(do.call(rbind, processed_data)))
    },
    
    search_submissions = function(query, author, method, start, end, score_min, score_max, subreddit, subreddit_id, limit, context, search_in, restricted, comments_min, comments_max) {
      params <- list(
        query = query,
        author = author,
        method = method,
        start = start,
        end = end,
        score_min = score_min,
        score_max = score_max,
        subreddit = subreddit,
        subreddit_id = subreddit_id,
        limit = limit,
        context = context,
        search_in = search_in,
        restricted = restricted,
        comments_min = comments_min,
        comments_max = comments_max
      )
      tryCatch({
        response <- .self$make_request('search/submissions', params)
        return(.self$process_submission_response(response))
      }, APIError = function(e) {
        cat("Error searching submissions:", e$message, "\n")
        return(data.frame())
      })
    },
    
    search_comments = function(query, author, method, start, end, score_min, score_max, subreddit, subreddit_id, limit, context, search_in, restricted) {
      params <- list(
        query = query,
        author = author,
        method = method,
        start = start,
        end = end,
        score_min = score_min,
        score_max = score_max,
        subreddit = subreddit,
        subreddit_id = subreddit_id,
        limit = limit,
        context = context,
        search_in = search_in,
        restricted = restricted
      )
      tryCatch({
        response <- .self$make_request('search/comments', params)
        return(.self$process_comment_response(response))
      }, APIError = function(e) {
        cat("Error searching comments:", e$message, "\n")
        return(data.frame())
      })
    }
  )
)
