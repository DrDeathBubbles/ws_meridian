import pandas as pd
import numpy as np

# Read the ticket sales data
ticket_data = pd.read_csv('TicketSales.csv')

# Convert Transaction Date to datetime format
ticket_data['Transaction_Date'] = pd.to_datetime(ticket_data['Transaction Date'], errors='coerce')

# Aggregate ticket sales by date
daily_ticket_sales = (ticket_data
    .dropna(subset=['Transaction_Date'])
    .groupby(ticket_data['Transaction_Date'].dt.date)
    .size()
    .reset_index(name='ticket_sales')
    .rename(columns={'Transaction_Date': 'date_id'})
)

# Convert date_id to string format for compatibility
daily_ticket_sales['date_id'] = daily_ticket_sales['date_id'].astype(str)

# Save the aggregated data
daily_ticket_sales.to_csv('daily_ticket_sales.csv', index=False)

print("Ticket sales data processed and saved to daily_ticket_sales.csv")