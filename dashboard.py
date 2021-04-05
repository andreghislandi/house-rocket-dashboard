import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

#map builders
import geopandas
import plotly.express as px
import folium
from streamlit_folium import folium_static
from folium.plugins   import MarkerCluster



st.set_page_config(layout="wide" )
st.title( 'House Rocket Company')
st.markdown( 'Welcome to House Rocket Data Analisys')


@st.cache(allow_output_mutation=True)
def get_data ( path ):
    data = pd.read_csv( path )

    return data

@st.cache(allow_output_mutation=True)
def get_geofile(url):
    geofile = geopandas.read_file(url)

    return geofile

def set_feature(data):
    #add new features
    data['price_sqft'] = data['price']/data['sqft_lot']

    return data

def overview_data( data ):
    #--------------
    #Data Overview
    #--------------

    f_attributes = st.sidebar.multiselect('Enter columns:', data.columns)
    # st.write( 'Your attributes:', f_attributes)
    f_zipcode = st.sidebar.multiselect('Enter zipcode', data['zipcode'].unique())
    # st.write( 'Your zipcodes:', f_zipcode)

    st.title('Data Overview')

    #attributes + zipcode = select columns and rows
    if (f_zipcode != []) & (f_attributes != []):
        data = data.loc[data['zipcode'].isin(f_zipcode), f_attributes]

    #zip only = select cols
    elif (f_zipcode != []) & (f_attributes == []):
        data = data.loc[data['zipcode'].isin(f_zipcode), :]

    #att only = select rows
    elif (f_zipcode == []) & (f_attributes != []):
        data = data.loc[:, f_attributes]
    # none selected = original data
    else:
        data = data.copy()

    st.dataframe(data.head())

    c1, c2 = st.beta_columns((1, 1))

    # Average Metrics
    df1 = data[['id','zipcode']].groupby('zipcode').count().reset_index()
    df2 = data[['price','zipcode']].groupby('zipcode').mean().reset_index()
    df3 = data[['sqft_living','zipcode']].groupby('zipcode').mean().reset_index()
    df4 = data[['price_sqft','zipcode']].groupby('zipcode').mean().reset_index()

    # Merge dfs
    m1 = pd.merge(df1,df2, on='zipcode', how='inner')
    m2 = pd.merge(m1,df3, on='zipcode', how='inner')
    df = pd.merge(m2,df4, on='zipcode', how='inner')

    df.columns = ['zipcode', 'total houses', 'price', 'sqft_living', 'price_sqft']

    c1.header('Average Metrics')
    c1.dataframe(df, width=1000, height=400)

    # Descriptive Statistics

    num_attributes = data.select_dtypes(include=['int64','float64'])
    mean = pd.DataFrame(num_attributes.apply(np.mean))
    median = pd.DataFrame(num_attributes.apply(np.median))
    std = pd.DataFrame(num_attributes.apply(np.std))
    min_ = pd.DataFrame(num_attributes.apply(np.min))
    max_ = pd.DataFrame(num_attributes.apply(np.max))

    df_desc = pd.concat([max_, min_, mean, median, std], axis=1).reset_index()
    df_desc.columns = ['attributes', 'max', 'min', 'avg', 'median', 'std']
    c2.header('Descriptive Stats')
    c2.dataframe(df_desc, width=1000, height=400)

    return None

def density_maps(data, geofile):
    #--------------
    #Density Maps
    #--------------

    st.title('Region Overview')

    c1, c2 = st.beta_columns((1, 1))

    c1.header('Portfolio Density')

    display_port_map = c1.checkbox('Display Portfolio Map (May take several seconds)')


    df = data

    # base map

    if display_port_map:

        density_map = folium.Map( location=[data['lat'].mean(), data['long'].mean()],
                                default_zoom_start=15
                                )
        marker_cluster = MarkerCluster().add_to(density_map)
        for name, row in df.iterrows():
            folium.Marker([row['lat'], row['long']],
            popup='Price ${0} on: {1}. Features: {2} sqft, {3} bedrooms, {4} bathrooms, year built: {5}'.format(
                row['price'], row['date'], row['sqft_living'], row['bedrooms'], row['bathrooms'],row['yr_built']
            )
            ).add_to(marker_cluster)

        with c1:
            folium_static(density_map)

    # price map

    c2.header('Price Density')

    display_price_map = c2.checkbox('Display Price Map')

    if display_price_map:

        df = data[['price','zipcode']].groupby('zipcode').mean().reset_index()
        df.columns = ['ZIP','price']

        geofile = geofile[geofile['ZIP'].isin(df['ZIP'].tolist())]

        price_density_map = folium.Map( location=[data['lat'].mean(), data['long'].mean()],
                                default_zoom_start=15
                                )

        price_density_map.choropleth( data = df,
                                    geo_data= geofile,
                                    columns=['ZIP','price'],
                                    key_on='feature.properties.ZIP',
                                    fill_color='YlOrRd',
                                    fill_opacity=0.4,
                                    line_opacity=0.2,
                                    legend_name='Avg Price'
                                    )

        with c2:
            folium_static(price_density_map)
    return None


def commercial(data):
    #--------------
    #Commercial Info
    #--------------

    st.title('Commercial Info')
    st.sidebar.title('Commercial options')

    # Filters

    ###load data
    data = get_data( path='kc_house_data.csv' )
    data['date'] = pd.to_datetime( data['date'] ).dt.strftime( '%Y-%m-%d' )

    # setup filters
    min_date = datetime.strptime( data['date'].min(), '%Y-%m-%d' )
    max_date = datetime.strptime( data['date'].max(), '%Y-%m-%d' )

    min_yr_built = int(data['yr_built'].min())
    max_yr_built = int(data['yr_built'].max())
    f_yr_built = st.sidebar.slider('Year built range:',
                        min_yr_built, 
                        max_yr_built, 
                        [min_yr_built, max_yr_built])

    min_date = datetime.strptime( data['date'].min(), '%Y-%m-%d' )
    max_date = datetime.strptime( data['date'].max(), '%Y-%m-%d' )
    f_date = st.sidebar.slider('Days range:',
                        min_date, 
                        max_date, 
                        max_date)

    # Avg price / yr
    st.header('Avg price per Year built')

    # apply filter / select data
    df = data[(data['yr_built'] > f_yr_built[0]) & (data['yr_built'] < f_yr_built[1])]
    df = df[['yr_built', 'price']].groupby('yr_built').mean().reset_index()

    # plot
    fig = px.line(df, x='yr_built', y='price')
    st.plotly_chart(fig, use_container_width=True)


    # Avg price / day

    st.header('Avg price per day')

    # apply filter / select data
    data['date'] = pd.to_datetime(data['date'])
    df = data[data['date'] < f_date]
    df = df[['date', 'price']].groupby('date').mean().reset_index()

    #plot
    fig = px.line(df, x='date', y='price')
    st.plotly_chart(fig, use_container_width=True)

    return None

def histograms(data):
    #--------------
    #Histogram
    #--------------

    st.header('Price Distribution')
    st.sidebar.subheader('Select Price')

    #filter
    min_price = int(data['price'].min())
    max_price = int(data['price'].max())
    avg_price = int(data['price'].mean())

    f_price = st.sidebar.slider('Max Price', min_price, max_price, avg_price)
    df = data[data['price'] < f_price]

    #plot hist
    fig = px.histogram( df, x='price', nbins=50)
    st.plotly_chart( fig, use_container_width=True)

    #--------------
    #Distribution by Features
    #--------------

    st.sidebar.title('Attributes Options')
    st.title('House Attributes')

    #filters
    f_bedrooms = st.sidebar.selectbox('Max num. of bedrooms',
                    sorted(set(data['bedrooms'].unique())))
    f_bathrooms = st.sidebar.selectbox('Max num. of bathrooms',
                    sorted(set(data['bathrooms'].unique())))
    f_floors = st.sidebar.selectbox('Max num. of floors',
                    sorted(set(data['floors'].unique())))
    f_waterview = st.sidebar.checkbox('Only waterfront')

    c1, c2 = st.beta_columns(2)

    #House per bedrooms
    c1.header('Houses per bedrooms')
    df = data[data['bedrooms'] <= f_bedrooms]
    fig = px.histogram(df, x='bedrooms', nbins=19)
    c1.plotly_chart( fig, use_container_width=True)

    #House per bathrooms
    c2.header('Houses per bathrooms')
    df = data[data['bathrooms'] <= f_bathrooms]
    fig = px.histogram(df, x='bathrooms', nbins=19)
    c2.plotly_chart( fig, use_container_width=True)

    #House per floors
    c1.header('Houses per floors')
    df = data[data['floors'] <= f_floors]
    fig = px.histogram(df, x='floors', nbins=19)
    c1.plotly_chart( fig, use_container_width=True)

    #Houses per waterview
    if f_waterview:
        df = data[data['waterfront'] == 1]

    else:
        df = data.copy()
    c2.header('Houses waterfront')
    fig = px.histogram(df, x='waterfront', nbins=19)
    c2.plotly_chart( fig, use_container_width=True)

    return None

def house_rocket_map(data):
    # plot map
    st.title('House Rocket Map')
    display_map = st.checkbox('Display Map')

    #filters
    price_min = int(data['price'].min())
    price_max = int(data['price'].max())
    price_avg = int(data['price'].mean())

    price_slider = st.slider(
        'Price Range',
        price_min,
        price_max,
        [price_min, price_avg]
    )

    if display_map:
        #select rows
        houses = data[(data['price'] > price_slider[0]) & (data['price'] < price_slider[1])][['id','lat',
                    'long',
                    'price']]

        #st.dataframe(houses)

        #draw map

        fig = px.scatter_mapbox(
            houses,
            lat='lat',
            lon='long',
            size='price',
            color_continuous_scale=px.colors.cyclical.IceFire,
            size_max=15,
            zoom=9,
            mapbox_style='open-street-map'
        )

        fig.update_layout(height=600, margin={'r':0,'t':0,'l':0,'b':0})
        st.plotly_chart(fig)

    return None

def hypothesis (data):

    st.title('Hypothesis Study')
    display_hypothesis = st.checkbox('Display Hypothesis Study')

    if display_hypothesis:
        #H1 - waterfront properties cost 30% more (avg)
        st.header('Hypothesis 01: Waterfront properties prices are 30% higher on average')
        mean_waterfront = data.loc[data['waterfront']==1]['price'].mean()
        mean_waterfront_not = data.loc[data['waterfront']==0]['price'].mean()
        st.write(f'Waterfront properties average price: **{mean_waterfront:.2f}**')
        st.write(f'Non-waterfront properties average price: **{mean_waterfront_not:.2f}**')
        st.write(f'Difference in %: {(mean_waterfront/mean_waterfront_not)*100:.2f}')
        st.markdown('Waterfront properties are actually over 300% higher on price')

        #H2 - properties built before 1955 are 50% cheaper (avg)
        st.header('Hypothesis 02: Properties built before 1955 are 50% cheaper, on average')
        mean_under_1955 = data.loc[data['yr_built'] < 1955]['price'].mean()
        mean_over_1955 = data.loc[data['yr_built'] >= 1955]['price'].mean()
        st.write(f'Properties built before 1955 average price: **{mean_under_1955:.2f}**')
        st.write(f'Properties built from 1955 on average price: **{mean_over_1955:.2f}**')
        st.write(f'Difference in %: {100-((mean_under_1955/mean_over_1955)*100):.2f}')
        st.markdown('Properties built before and after 1955 share a similar price average')

        #H3 - properties without basement have a total lot size 50% bigger than properties with basement
        st.header('Hypothesis 03: Properties without basement have a total lot size 50% bigger than properties with basement, on average')
        lot_size_no_base = data.loc[data['sqft_basement'] == 0]['sqft_lot'].mean()
        lot_size_with_base = data.loc[data['sqft_basement'] > 0]['sqft_lot'].mean()
        st.write(f'Properties without basement average lot size: **{lot_size_no_base:.2f}**')
        st.write(f'Properties with basement average lot size: **{lot_size_with_base:.2f}**')
        st.write(f'Difference in %: {100-((lot_size_with_base/lot_size_no_base)*100):.2f}')
        st.markdown('Properties without basement are on average 18% bigger on lot size than without basement.')

        #H4 - properties prices YoY increase is 10%
        st.header('Hypothesis 04: Properties prices YoY (Year over Year) increase is 10%')
        data['date'] = pd.to_datetime( data['date'] ).dt.strftime( '%Y-%m-%d' )
        prices_may_2014 = data.loc[(data['date'] > '2014-05-01') & (data['date'] < '2014-06-01')]['price'].mean()
        prices_may_2015 = data.loc[(data['date'] > '2015-05-01') & (data['date'] < '2015-06-01')]['price'].mean()
        st.write(f'Average price on May 2014: **{prices_may_2014:.2f}**')
        st.write(f'Average price on May 2015: **{prices_may_2015:.2f}**')
        st.write(f'Difference in %: {100-((prices_may_2014/prices_may_2015)*100):.2f}')
        st.markdown('Year over Year increase in properties prices is only 1% (From May 2014 to May 2015)')

        #H5 - properties with 3 bathrooms prices grow 15% MoM
        st.header('Hypothesis 05: Properties with 3 bathrooms prices MoM (Month over Month) increase is 15%')
        data['date'] = pd.to_datetime( data['date'], format='%Y-%m-%d' )
        date_monthly = pd.date_range("2014-04-30", periods=14, freq="M")
        percent_monthly = []
        for i in range(len(date_monthly)-1):
            percent_monthly.append(data.loc[(data['date'] > date_monthly[i]) & (data['date'] < date_monthly[i+1]) & (data['bathrooms']==3)]['price'].mean())
        for i in range(len(percent_monthly)-1):
            percent_monthly[i] = (100-((percent_monthly[i]/percent_monthly[i+1])*100))
        percent_monthly.pop()

        st.write(f'Average MoM in 12 months: **{sum(percent_monthly)/len(percent_monthly):.2f}** %')
        
        fig = px.line(x=date_monthly[1:13], y=percent_monthly)
        fig.update_layout(xaxis_title='Months',
                    yaxis_title='MoM percentage')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('Month over Month change in properties prices with 3 bathrooms is only -0.32% average on a 12 month period')

        #H6 - 50% of the waterfront properties were sold during the summer(june, jul, aug)
        st.header('Hypothesis 06: 50% of the waterfront properties were sold during the summer(june, jul, aug)')
        df = data.loc[data['waterfront']==1][['id','date']]
        df['date'] = pd.to_datetime( df['date'], format='%Y-%m-%d' )
        summer_sales = df.loc[(df['date'] >= '2014-06-01') & (df['date'] < '2014-09-01')]['id'].count()
        winter_sales = df.loc[(df['date'] > '2014-05-01') & (df['date'] < '2014-06-01') | (df['date'] >= '2014-09-01')]['id'].count()
        st.write(f'Waterfront Houses sold on Summer: **{summer_sales}**')
        st.write(f'Waterfront Houses sold on rest of year: **{winter_sales}**')
        st.write(f'% to total: {(summer_sales/(summer_sales+winter_sales)*100):.2f}')
        st.markdown('Approximately 30% of waterfront properties were sold during the summer')

def houses_to_buy (data):

    st.title('Suggested Houses to Buy')
    st.markdown('- Below Median price for the region \n - Waterfront \n - Good Condition')
    df = data.loc[(data['waterfront']==1) & (data['condition'] >=4)][['id','price','zipcode','lat','long','condition','bedrooms','bathrooms','floors','sqft_lot','yr_built']]
    df1 = df[['price','zipcode']].groupby('zipcode').median().reset_index()
    df1.columns = ['zipcode','zip_median']
    df = pd.merge(df, df1, on='zipcode', how='inner')
    df['status'] = 'N'
    for i in range(len(df)):
        if df.loc[i]['price'] < df.loc[i]['zip_median']:
            df.status[i] = 'Buy'
        if df.loc[i]['price'] >= df.loc[i]['zip_median']:
            df.status[i] = 'No Buy'
    st.write(df)

    #draw map
    st.header('Houses to buy: Map')
    houses_buy = df[df['status']=='Buy']
    fig = px.scatter_mapbox(
            houses_buy,
            lat='lat',
            lon='long',
            size='price',
            color='condition',
            hover_name='id',
            hover_data=['price','condition','bedrooms','bathrooms','floors','sqft_lot','yr_built'],
            color_continuous_scale=px.colors.sequential.Aggrnyl,
            size_max=15,
            zoom=9,
    )

    fig.update_layout(
    mapbox_style="white-bg",
    mapbox_layers=[
        {
            "below": 'traces',
            "sourcetype": "raster",
            "sourceattribution": "United States Geological Survey",
            "source": [
                "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
            ]
        }
      ])
    fig.update_layout(height=600, margin={'r':0,'t':0,'l':0,'b':0})
    st.plotly_chart(fig)

if __name__ == '__main__':
    #ETL
    #data extraction
    path = 'kc_house_data.csv'
    url = 'https://data-seattlecitygis.opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'

    data = get_data( path )
    geofile = get_geofile(url)

    #transformation
    data = set_feature(data)

    overview_data(data)

    density_maps(data, geofile)

    house_rocket_map(data)

    commercial(data)

    histograms(data)

    hypothesis(data)

    houses_to_buy(data)