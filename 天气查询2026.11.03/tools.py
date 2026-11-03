
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import requests

# ==================================1.配置访问api================================
my_api_key = "d5e8ee7e9c3340f8ba091a7ac9895ba1"
my_host = "ht5b682k53.re.qweatherapi.com"

#===================================2.构造工具列表===============================
#2.1构造查询城市location经纬度
def get_city_location(city:str):
    """"
    构造工具函数的子函数：
    查询指定城市的经纬度坐标。
    """
    # 配置访问api
    QW_API_KEY = my_api_key
    HOST = my_host
    LOCATION = city
    
    #网站查询城市经纬度 

    url = f"https://{HOST}/geo/v2/city/lookup"
    headers = {"X-QW-Api-Key": QW_API_KEY}
    params = {"location": LOCATION}
        
    res = requests.get(url, headers=headers, params=params, timeout=10)
    location_id = res.json()["location"][0]["id"]
    return location_id
        
    
#2.2构造查询天气工具
class Weather(BaseModel):
    city: str = Field(description="城市名")
    # location:str = Field(description="城市经纬度")

@tool(args_schema=Weather)
def get_weather(city:str):
    """查询指定城市的当前3日内天气。"""
    
    #第一步通过城市名查询经纬度
    location = get_city_location(city)
    
    #第二步：通过城市location经纬度，查询天气情况
    #  控制台的API Key 和你的专属Host 
    QW_API_KEY = my_api_key
    HOST = my_host
    LOCATION = location
    # 

    url = f"https://{HOST}/v7/weather/3d"
    headers = {"X-QW-Api-Key": QW_API_KEY}
    params = {"location": LOCATION}
           
    res = requests.get(url, headers=headers, params=params, timeout=10)

    daily_datas = f"{city}天气情况：\n"
    for daily_dict in res.json()["daily"]:
        daily_data = daily_dict["fxDate"] + f"天气{daily_dict['textDay']}," + f"最高温度{daily_dict['tempMax']}摄氏度," + f"最低温度{daily_dict['tempMin']}摄氏度!\n"
        daily_datas += daily_data
    return daily_datas

