# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 16:42:05 2018

@author: Evan_He
"""

from urllib.parse import urlparse
import time
import re
import urllib.request
from urllib import robotparser
from urllib.error import URLError, HTTPError, ContentTooShortError
import requests
import requests_cache
import csv
from lxml.html import fromstring
from datetime import timedelta,datetime
import os
from urllib.parse import urlsplit
import zlib
import json
from redis import StrictRedis
import socket
import threading

SLEEP_TIME = 5


class Throttle:

    def __init__(self, delay):
        # amount of delay between downloads for each domain
        self.delay = delay
        # timestamp of when a domain was last accessed
        self.domains = {}

    def wait(self, url):
        domain = urlparse(url).netloc
        last_accessed = self.domains.get(domain)

        if self.delay > 0 and last_accessed is not None:
            sleep_secs = self.delay - (time.time() - last_accessed)
            if sleep_secs > 0:
                time.sleep(sleep_secs)
        self.domains[domain] = time.time()

def download_re(url, num_retries=2, charset='utf-8'):
    print('Downloading:', url)
    request = urllib.request.Request(url)
    try:
        resp = urllib.request.urlopen(request)
        cs = resp.headers.get_content_charset()
        if not cs:
            cs = charset
        html = resp.read().decode(cs)
    except (URLError, HTTPError, ContentTooShortError) as e:
        print('Download error:', e.reason)
        html = None
        if num_retries > 0:
            if hasattr(e, 'code') and 500 <= e.code < 600:
                # recursively retry 5xx HTTP errors
                return download_re(url, num_retries - 1)
    return html

class CsvCallback:
    #回调类存储
    def __init__(self):
        self.writer = csv.writer(open('./resold_apartment.csv', 'w',encoding='utf-8', newline=''))
        self.fields = ['链家id','小区名称','别墅','布局','面积','房屋朝向','装修','电梯','区域','地址','商区','备注','总价','单价']      
        self.writer.writerow(self.fields)

    def __call__(self, html,zone):
        self.writer = csv.writer(open('./ershoufang.csv', 'a',encoding='utf-8', newline=''))
        all_rows = []
        try:
            housecode =str(re.compile("""data-housecode=[",'](.*?)[",']""", re.IGNORECASE|re.DOTALL).findall(html)[0])
        except:
            housecode = ''
        all_rows.append(housecode)    
        html1 = fromstring(html)
        try:
            house_name =  html1.xpath('//div[@class="address"]/div[@class="houseInfo"]/a')[0].text_content()
        except:
            house_name = ''
        all_rows.append(house_name)
        room_info = html1.xpath('//div[@class="address"]/div[@class="houseInfo"]/text()')[0].strip()
        room_info = room_info.split('|')
        if '别墅' in room_info[1]:
            all_rows.append(room_info[1])
            all_rows.append(room_info[2])
            all_rows.append(room_info[3])
            all_rows.append(room_info[4])
            all_rows.append(room_info[5])
            try:
                all_rows.append(room_info[6])
            except:
                all_rows.append('无')
        else:
            all_rows.append('否')
            all_rows.append(room_info[1])
            all_rows.append(room_info[2])
            all_rows.append(room_info[3])
            try:
                all_rows.append(room_info[4])
            except:
                all_rows.append('')
            try:
                all_rows.append(room_info[5])
            except:
                all_rows.append('无')
    
        all_rows.append(zone)
        try:
            address =  html1.xpath('//div[@class="flood"]/div[@class="positionInfo"]/text()')[0]
        except:
            address = ''
        all_rows.append(address)
        
        try:
            shop =  html1.xpath('//div[@class="flood"]/div[@class="positionInfo"]/a')[0].text_content()
        except:
            shop = ''
        all_rows.append(shop)               
        try:
            note2 = html1.xpath('//div[@class="tag"]/span[@class="subway"]')[0].text_content()
        except:
            note2 = ''
        all_rows.append(note2)
        try:
            total_price = html1.xpath('//div[@class="priceInfo"]/div[@class="totalPrice"]/span')[0].text_content()
        except:
            total_price = ''
        all_rows.append(total_price) 
        try:
            price = html1.xpath('//div[@class="priceInfo"]/div[2]/span')[0].text_content()
        except:
            price = ''
        all_rows.append(price) 
          
         
        self.writer.writerow(all_rows)
        

def get_robots_parser(robots_url):
    " Return the robots parser object using the robots_url "
    try:
        rp = robotparser.RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return rp
    except Exception as e:
        print('Error finding robots_url:', robots_url, e)
            
            
class Downloader_request_cache:
    def __init__(self, delay=5,timeout=60):
        self.throttle = Throttle(delay)
        self.num_retries = None  # we will set this per request
        self.timeout = timeout

    def __call__(self, url, num_retries=2):
        self.num_retries = num_retries
        result = self.download(url)
        return result['html']

    def make_throttle_hook(self, throttle=None):
        def hook(response, *args, **kwargs):
            """ see requests hook documentation for more information"""
            if not getattr(response, 'from_cache', False):
                throttle.wait(response.url)
                print('Downloading1:', response.url)
            else:
                print('Returning from cache:', response.url)
            return response
        return hook

    def download(self, url):
        session = requests_cache.CachedSession()
        session.hooks = {'response': self.make_throttle_hook(self.throttle)}

        try:
            resp = session.get(url, timeout=self.timeout)
            html = resp.text
            if resp.status_code >= 400:
                print('Download error:', resp.text)
                html = None
                if self.num_retries and 500 <= resp.status_code < 600:
                    # recursively retry 5xx HTTP errors
                    self.num_retries -= 1
                    return self.download(url)
        except requests.exceptions.RequestException as e:
            print('Download error:', e)
            return {'html': None, 'code': 500}
        return {'html': html, 'code': resp.status_code}

def link_crawler_threaded_crawler(start_url, delay=3, max_threads=5,num_retries=2,
                                 expires=timedelta(days=30),scraper_callback=None):

    requests_cache.install_cache(backend='', expire_after=expires)
    D = Downloader_request_cache(delay=delay)
    def process_queue():
        while start_url:
            url,zone = start_url.pop()
            if not url or 'https' not in url:
                continue
            html = D(url,num_retries=num_retries)
            if not html:
                continue
            
            re_house = re.compile("""<ul class=[",']sellListContent(.*?)</ul>""", re.IGNORECASE|re.DOTALL)  
            re_houselists = re.compile("""<li class=[",']clear LOGCLICKDATA[",'] >(.*?)</li>""",re.IGNORECASE|re.DOTALL) 
    
            houselist = re_houselists.findall(re_house.findall(html)[0])
            if scraper_callback:
                for house in houselist:
                    scraper_callback(house,zone)
            else:
                print('Blocked by robots.txt:', url)
    threads = []
    while threads or start_url:
        for thread in threads:
            if not thread.is_alive():
                threads.remove(thread)
        while len(threads) < max_threads and start_url:
            # can start some more threads
            thread = threading.Thread(target=process_queue)
            thread.setDaemon(True)  # set daemon so main thread can exit w/ ctrl-c
            thread.start() 
            threads.append(thread)
        for thread in threads:
            thread.join()

        time.sleep(SLEEP_TIME)


def find_url_count(url):
    html = download_re(url+'/resold_apartment/')
    re_zone_all = re.compile("""<div\s+data-role=[",']ershoufang[",']\s+>\s+<div>(.*?)\s+</div>\s+</div>""", re.IGNORECASE|re.DOTALL)
    re_zone_lists = re.compile('<a href=["](.*?)["]  title', re.IGNORECASE|re.DOTALL)
    zone_lists=re_zone_lists.findall(re_zone_all.findall(html)[0])
    url_count_list = []
    zone_list_name = ['锦江','青羊','武侯','高新','成华','金牛','天府新区','高新西','双流','温江','郫都','龙泉驿','新都','天府新区南区']
    for i in range(len(zone_lists)):
        for lc in ['lc1','lc2','lc3']:
            zone_url = url+ zone_lists[i]
            zone_url_lc = zone_url+lc
            zone_html = download_re(zone_url_lc)
            page_find=re.compile("""page-data=[",']{"totalPage":(.*?),[",']curPage[",']:1}""",re.IGNORECASE|re.DOTALL)
            try:
                page_count = int(page_find.findall(zone_html)[0])
            except IndexError:
                continue
            url_count_list.append((zone_list_name[i],lc,zone_url,page_count))
        time.sleep(1.5)
    return url_count_list

def scrap_url_lists(url):
    url_count_list = find_url_count(url)
    url_lists = []
    for zone_list_name,lc,zone_url,count in url_count_list:
        for i in range(1,count+1):
            url_lists.append((zone_url+'pg'+str(i)+'/'+lc,zone_list_name))
    return url_lists
        

#url  = 'https://cd.lianjia.com'
#urlists = scrap_url_lists(url)
b = open(r"./url_list.txt", "r",encoding='UTF-8')
out = b.read()
out = json.loads(out)
#
#
#
scraper_callback = CsvCallback()
link_crawler_threaded_crawler(out, delay=3, max_threads=5,num_retries=1,
                                 expires=timedelta(days=30),scraper_callback=scraper_callback)

#url = 'https://cd.lianjia.com/ershoufang/tianfuxinqunanqu/lc3/'
#html = download_re(url)
#re_house = re.compile("""<ul class=[",']sellListContent[",'] log-mod="list">(.*?)</ul>""", re.IGNORECASE|re.DOTALL)  
#re_houselists = re.compile("""<li class=[",']clear LOGCLICKDATA[",'] >(.*?)</li>""",re.IGNORECASE|re.DOTALL) 
#    
#houselist = re_houselists.findall(re_house.findall(html)[0])[0]
    


    
    