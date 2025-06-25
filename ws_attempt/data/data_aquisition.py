import requests
import pandas as pd
import matplotlib.pyplot as plt
import os

metabase_token = os.environ.get('METABASE_TOKEN')


def get_card_data(question_id, metabase_token):
    res = requests.post(f'https://metabase.cilabs.com/api/card/{question_id}/query/json', 
                  headers = {'Content-Type': 'application/json',
                            'X-Metabase-Session': metabase_token 
                            }
                )
    out_data = pd.DataFrame(res.json())
    return out_data


def local_import_data():
    #marketing_spend = pd.read_csv('./data/raw_data/advertising_raw.csv')
    #ticket_sales = pd.read_csv('./data/raw_data/ticket_sales_raw.csv')
    marketing_spend = pd.read_csv('/Users/aaronmeagher/Work/google_meridian/google/ws_attempt/data/raw_data/advertising_raw.csv')
    ticket_sales = pd.read_csv('/Users/aaronmeagher/Work/google_meridian/google/ws_attempt/data/raw_data/ticket_sales_raw.csv')
    #marketing_spend = pd.read_csv('/home/ubuntu/work/marketing/marketing_analytics/functionalised/data/raw_data/advertising_raw.csv')
    #ticket_sales = pd.read_csv('/home/ubuntu/work/marketing/marketing_analytics/functionalised/data/raw_data/ticket_sales_raw.csv')
    data = [marketing_spend, ticket_sales]
    data = process_data(data)
    
    marketing_spend = pd.concat(data[:-1], axis = 1)
    marketing_spend['date_week']= marketing_spend.index
    marketing_spend['date_week']=marketing_spend.index
    
    y = data[-1]
    common_dates = y.index.intersection(marketing_spend.index)
    marketing_spend_common =  marketing_spend.loc[common_dates]
    marketing_spend_common.fillna(0, inplace=True)
    y_common = y.loc[common_dates]
    return [y_common, marketing_spend_common]
    

def process_data(data):
    marketing_spend, ticket_sales = data

    ticket_sales['Date ID'] = pd.to_datetime(ticket_sales['Date ID'])
    min_date = min(ticket_sales['Date ID'])
    max_date = max(ticket_sales['Date ID'])
    overall_date_range = pd.date_range(min_date, max_date)
    ticket_sales.set_index('Date ID', inplace=True)
    ticket_sales = ticket_sales['Net Paid'].groupby(ticket_sales.index).sum()

    marketing_spend['date_id'] = pd.to_datetime(marketing_spend['date_id'])
    min_date = min(marketing_spend['date_id'])
    max_date = max(marketing_spend['date_id'])
    overall_date_range = pd.date_range(min_date, max_date)
    overall_date_range
    marketing_spend.index
    marketing_spend['date_id'] = pd.to_datetime(marketing_spend['date_id'])
    facebook = marketing_spend[marketing_spend['marketing_platform_name']=='FacebookAds'].add_prefix('facebook_')
    linkedin = marketing_spend[marketing_spend['marketing_platform_name']=='LinkedInAds'].add_prefix('linkedin_')
    google = marketing_spend[marketing_spend['marketing_platform_name']=='GoogleAds'].add_prefix('google_')
    bing = marketing_spend[marketing_spend['marketing_platform_name']=='BingAds'].add_prefix('bing_')
    ticktock = marketing_spend[marketing_spend['marketing_platform_name']=='TikTok'].add_prefix('ticktock_')
    twitter = marketing_spend[marketing_spend['marketing_platform_name']=='TwitterAds'].add_prefix('twitter_')    
    instagram = marketing_spend[marketing_spend['marketing_platform_name']=='Instagram'].add_prefix('instagram_')
    reddit = marketing_spend[marketing_spend['marketing_platform_name']=='Reddit'].add_prefix('reddit_')


    #facebook['facebook_date_id'] = pd.to_datetime(facebook['facebook_date_id'])
    facebook = facebook.set_index(facebook['facebook_date_id'])
    facebook = facebook['facebook_campaign_spend_eur'].groupby(facebook.index).sum()

    linkedin = linkedin.set_index(linkedin['linkedin_date_id'])
    linkedin = linkedin['linkedin_campaign_spend_eur'].groupby(linkedin.index).sum()    

    google = google.set_index(google['google_date_id']) 
    google = google['google_campaign_spend_eur'].groupby(google.index).sum()

    bing = bing.set_index(bing['bing_date_id'])
    bing = bing['bing_campaign_spend_eur'].groupby(bing.index).sum()

    ticktock = ticktock.set_index(ticktock['ticktock_date_id'])
    ticktock = ticktock['ticktock_campaign_spend_eur'].groupby(ticktock.index).sum()

    twitter = twitter.set_index(twitter['twitter_date_id'])
    twitter = twitter['twitter_campaign_spend_eur'].groupby(twitter.index).sum()

    instagram = instagram.set_index(instagram['instagram_date_id'])
    instagram = instagram['instagram_campaign_spend_eur'].groupby(instagram.index).sum()

    reddit = reddit.set_index(reddit['reddit_date_id'])
    reddit = reddit['reddit_campaign_spend_eur'].groupby(reddit.index).sum()
    #This is the bit here - I am selecting only the reddit_campaign_spend and not the impressions. See how easily this can be updated - piece by piece:w
    
    
    #return [google, facebook, twitter, ticktock, instagram, reddit, ticket_sales]
    return [google, facebook, twitter, ticktock, instagram, reddit, bing, linkedin, ticket_sales]


def plot_data(data):
    google, facebook, twitter, ticktock, instagram, reddit = data[:-1]
    fig, ax = plt.subplots(figsize=(10,8))
    google.plot(ax = ax, label='Google')
    facebook.plot(ax = ax, label = 'Facebook')
    twitter.plot(ax = ax, label = 'Twitter')
    ticktock.plot(ax = ax, label = 'Tiktok')
    #twitter.plot(ax = ax, label = 'Twitter')
    instagram.plot(ax = ax, label = 'Instagram')
    reddit.plot(ax = ax, label = 'Reddit')
    plt.xlabel('Date')
    plt.ylabel('Spend')
    plt.legend()



def get_raw_data(save= True):
    #marketing_spend = get_card_data(50357, metabase_token)
    marketing_spend = get_card_data(51475, metabase_token) #for testing
    ticket_sales = get_card_data(51178, metabase_token)
    if save == True:
        marketing_spend.to_csv('./data/raw_data/advertising_raw.csv')
        ticket_sales.to_csv('./data/raw_data/ticket_sales_raw.csv')
    return [marketing_spend, ticket_sales]    

def process_data(data):
    marketing_spend, ticket_sales = data

    ticket_sales['Date ID'] = pd.to_datetime(ticket_sales['Date ID'])
    min_date = min(ticket_sales['Date ID'])
    max_date = max(ticket_sales['Date ID'])
    overall_date_range = pd.date_range(min_date, max_date)
    ticket_sales.set_index('Date ID', inplace=True)
    ticket_sales = ticket_sales['Net Paid'].groupby(ticket_sales.index).sum()

    marketing_spend['date_id'] = pd.to_datetime(marketing_spend['date_id'])
    min_date = min(marketing_spend['date_id'])
    max_date = max(marketing_spend['date_id'])
    overall_date_range = pd.date_range(min_date, max_date)
    overall_date_range
    marketing_spend.index
    marketing_spend['date_id'] = pd.to_datetime(marketing_spend['date_id'])
    facebook = marketing_spend[marketing_spend['marketing_platform_name']=='FacebookAds'].add_prefix('facebook_')
    linkedin = marketing_spend[marketing_spend['marketing_platform_name']=='LinkedInAds'].add_prefix('linkedin_')
    google = marketing_spend[marketing_spend['marketing_platform_name']=='GoogleAds'].add_prefix('google_')
    bing = marketing_spend[marketing_spend['marketing_platform_name']=='BingAds'].add_prefix('bing_')
    ticktock = marketing_spend[marketing_spend['marketing_platform_name']=='TikTok'].add_prefix('ticktock_')
    twitter = marketing_spend[marketing_spend['marketing_platform_name']=='TwitterAds'].add_prefix('twitter_')    
    instagram = marketing_spend[marketing_spend['marketing_platform_name']=='Instagram'].add_prefix('instagram_')
    reddit = marketing_spend[marketing_spend['marketing_platform_name']=='Reddit'].add_prefix('reddit_')


    #facebook['facebook_date_id'] = pd.to_datetime(facebook['facebook_date_id'])
    facebook = facebook.set_index(facebook['facebook_date_id'])
    facebook = facebook['facebook_campaign_spend_eur'].groupby(facebook.index).sum()

    linkedin = linkedin.set_index(linkedin['linkedin_date_id'])
    linkedin = linkedin['linkedin_campaign_spend_eur'].groupby(linkedin.index).sum()    

    google = google.set_index(google['google_date_id']) 
    google = google['google_campaign_spend_eur'].groupby(google.index).sum()

    bing = bing.set_index(bing['bing_date_id'])
    bing = bing['bing_campaign_spend_eur'].groupby(bing.index).sum()

    ticktock = ticktock.set_index(ticktock['ticktock_date_id'])
    ticktock = ticktock['ticktock_campaign_spend_eur'].groupby(ticktock.index).sum()

    twitter = twitter.set_index(twitter['twitter_date_id'])
    twitter = twitter['twitter_campaign_spend_eur'].groupby(twitter.index).sum()

    instagram = instagram.set_index(instagram['instagram_date_id'])
    instagram = instagram['instagram_campaign_spend_eur'].groupby(instagram.index).sum()

    reddit = reddit.set_index(reddit['reddit_date_id'])
    reddit = reddit['reddit_campaign_spend_eur'].groupby(reddit.index).sum()
    
    return [google, facebook, twitter, ticktock, instagram, reddit, ticket_sales]


if '__name__' == '__main__':
    raw_data = get_raw_data()
    processed_data = process_data(raw_data)
    plot_data(processed_data)