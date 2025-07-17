library(dplyr)
library(lubridate)

# Read the ticket sales data
ticket_data <- read.csv("TicketSales.csv")

# Convert Transaction Date to Date format
ticket_data$Transaction_Date <- as.Date(ticket_data$Transaction_Date)

# Aggregate ticket sales by date
daily_ticket_sales <- ticket_data %>%
  filter(!is.na(Transaction_Date)) %>%
  group_by(Transaction_Date) %>%
  summarise(ticket_sales = n()) %>%
  rename(date_id = Transaction_Date)

# Save the aggregated data
write.csv(daily_ticket_sales, "daily_ticket_sales.csv", row.names = FALSE)

print("Ticket sales data processed and saved to daily_ticket_sales.csv")