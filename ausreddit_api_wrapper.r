# r Ausreddit API wrapper

library(httr)
library(jsonlite)

base_url <- "https://ausreddit.digitialobservatory.net.au/api/v1"

# authenticate


#get requests

list_subreddits <- function(metadata = FALSE, as_dataframe = TRUE) {
  url <- paste0(base_url, "/subreddits")
  
  # Add a query parameter for metadata if needed
  query <- list()
  if (metadata) {
    query$include_metadata <- "true"
  }
  
  response <- GET(url, query = query)
  
  # Check for errors
  if (http_error(response)) {
    stop(
      sprintf(
        "API request failed [%s]\n%s",
        status_code(response),
        content(response, "text")
      ),
      call. = FALSE
    )
  }
  
  # Parse the JSON content
  parsed_content <- jsonlite::fromJSON(content(response, "text"), simplifyVector = FALSE)
  
  # Extract the relevant data (assuming it's in a 'data' field)
  subreddits <- parsed_content$data$children
  
  # If as_dataframe is TRUE, convert to dataframe
  if (as_dataframe) {
    subreddits_df <- do.call(rbind, lapply(subreddits, function(x) {
      data.frame(
        id = x$data$id,
        display_name = x$data$display_name,
        title = x$data$title,
        description = x$data$description,
        public_description = x$data$public_description,
        created_utc = as.POSIXct(x$data$created_utc, origin = "1970-01-01"),
        subscribers = x$data$subscribers,
        over18 = x$data$over18,
        url = x$data$url,
        banner_img = x$data$banner_img,
        icon_img = x$data$icon_img,
        community_icon = x$data$community_icon,
        lang = x$data$lang,
        stringsAsFactors = FALSE
      )
    }))
    return(subreddits_df)
  } else {
    return(subreddits)
  }
}

