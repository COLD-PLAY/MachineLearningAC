import requests
import re

date0 = ['201708', '201608', '', '', '', '', '']
date1 = ['201708', '201608', '20158', '20148', '20138', '20128', '20118']

url = 'http://tianqi.2345.com/t/wea_history/js/%s/56294_%s.js'
file = open('weatherTraining.txt', 'a')

for i in range(7):
	response = requests.get(url % (date0[i], date1[i]))

	# print(response.status_code)

	pattern = "ymd:'(.*?)',bWendu:'(.*?)℃',yWendu:'(.*?)℃',tianqi:'(.*?)'"

	weathers = re.findall(pattern, response.text, re.S)

	for weather in weathers:
		print(weather)

		flag = '0'
		if '雨' in weather[3]:
			flag = '1'

		file.write(weather[1] + '\t' + weather[2] + '\t' + flag + '\n')