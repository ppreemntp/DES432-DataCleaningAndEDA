# DES423 Project 1: Data Cleaning
# Please use pip install command to install pandas library before running this code.

import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import kagglehub
import numpy as np

# Download dataset from kaggle source and print the path to the dataset files
path = kagglehub.dataset_download('dgomonov/new-york-city-airbnb-open-data')
print("Path to dataset files:", path)
#airbnb = pd.read_csv(path)

# Check current directory and assign dataset to "airbnb" variable
airbnb = pd.read_csv("data/AB_NYC_2019.csv")

# Explore the dataset
print(airbnb.shape)
print(airbnb.dtypes)
#airbnb.info()

print("<--------------------------------------------------------------->")
print("<<<<< Check for duplication rows in the dataset >>>>>") # Check for duplication rows
duplicates = airbnb.duplicated().sum()
print("Number of duplicate rows:", duplicates) # No duplication which is good
print("<--------------------------------------------------------------->")


print("<<<<< Check for duplication IDs in the dataset >>>>>") #Check for duplication in the "id" column (id ห้ามซ้ำ)
print("Duplicate listing IDs:", airbnb.duplicated(subset="id").sum())
print("Duplicate host IDs:", airbnb.duplicated(subset="host_id").sum()) # Host ID ซ้ำได้เพราะ หนึ่งคนปล่อยเช่าได้หลายที่
print("<--------------------------------------------------------------->")


print("<<<<< Check for NaN values >>>>>") # Check for null values in the dataset
print(airbnb.isnull().sum())
# There are multiple missing values which are name, host_name, last_review, reviews_per_months. For name and host_name can be filled depends on wether analyst use it or not
# But for last_review and reviews_per_month. Missing exactly 10052 values which must be a relation. Our hypothesis is row with these missing values have no reviews
#Check is its the same rows that two colums have missing values
print(airbnb[
    (airbnb["last_review"].isnull()) &
    (airbnb["reviews_per_month"].isnull())
].shape) # Missing Values ทั้งสองคอลลัมตรงกัน
# เชคว่าจำนวนรีวิวเป็น 0 รึเปล่า มันเลย null (เชคทั้งสอง column)
print(airbnb[
    (airbnb["number_of_reviews"] == 0) &
    (airbnb["reviews_per_month"].isnull())
].shape)
print(airbnb[
    (airbnb["number_of_reviews"] == 0) &
    (airbnb["last_review"].isnull())
].shape)
#ใช่ conclude ได้เลยว่า missing values ในสอง column นี้คือยังไม่เคยมีคนรีวิวเลย เราสามารถเติม null values พวกนี้เป็นค่าเฉลี่ยเพื่อ analyst ในอนาคตจะได้หาค่าเฉลี่ยของทั้ง dataset ได้
print("<--------------------------------------------------------------->")

print("<<<<< Check Rediculous Pricing >>>>>")# Check for rediculously low or high prices ราคาควรเหมาะสม
print("Min price:", airbnb["price"].min())
print("Max price:", airbnb["price"].max())
# มีคนตั้งราคาเป็น 0 เราก็เรียกดูเพิ่ม
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)
print(airbnb[airbnb["price"] == 0]) # มี 11 ที่ที่ตั้งราคาเป็น 0 น่าจะดรอปทิ้งได้เพราะจำนวนน้อยและน่าจะไม่กระทบต่อการวิเคราะห์ในอนาคตมาก
print(airbnb.sort_values("price", ascending=False).head(20)) # เช็คราคาสูง อาจจะเป็นที่พัก Luxury หรือ input error ถ้าเป็น input error ก็ควรจะดรอปทิ้งแต่ถ้าเป็นที่พัก Luxury ก็อาจจะเก็บไว้ก็ได้เพราะมันก็มีอยู่จริงๆ

print("<--------------------------------------------------------------->")


print("<<<<< Check Minimum Nights >>>>>") #Check for night number จำนวนคืนที่จองได้ไม่ควรเป็น 0 หรือ เยอะเว่อร์เกินไป
print(airbnb["minimum_nights"].describe()) #มีคนตั้งขั้นต่ำเป็น 1250 คืนซึ่งอาจจะปล่อยเช่าระยะยาวอะไรงี้ป้ะ
print(airbnb.sort_values("minimum_nights", ascending=False).head(10))
print("Hosting with minimum nights more than 100: ")
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)
print(airbnb[airbnb["minimum_nights"] > 100]) # อาจจะปล่อยเช่าระยะยาวรึเปล่า แต่มีอยู่ 174 ที่ที่มันตั้งขั้นต่ำเป็น 100 คืนขึ้นไป
print("<--------------------------------------------------------------->")


print("<<<<< Check Minimum Nights >>>>>") # Check for availability ควรเป็น 0-365 วัน
print(airbnb["availability_365"].describe()) #ถูกแล้ว ไม่มีที่ไหน ปีนึงว่างเกิน 365 วัน
print("<--------------------------------------------------------------->")


print("<<<<< Check Pricing Distribution >>>>>")# Check price distribution
print(airbnb["price"].describe())
sns.histplot(airbnb["price"], bins=200)
plt.xlim(0, 1400)
plt.xticks(np.arange(0, 1401, 100))
plt.title("Distribution of Price (Zoomed)")
print("Mean price:", airbnb["price"].mean())
print("Median price:", airbnb["price"].median())
plt.show() # ราคาห้องต่อคืน right skewed มากเพราะมีคนตั้งราคาปัญญาอ่อนอยู่เช่น 0 ดอล ถ้าเราทำการแก้ไข้คอลลัม price โดยการดรอปทิ้งที่ตั้งราคาเป็น 0 ดอลไปแล้ว เราก็จะได้ distribution ที่ดูสมเหตุสมผลมากขึ้น
#เพิ่ม boxplot เพื่อน compare
sns.boxplot(x=airbnb["price"])
plt.xlim(0, 2000)
plt.title("Boxplot of Price")
plt.show()
print("<--------------------------------------------------------------->")


print("<<<<< Check Minimum Nights Distribution >>>>>")
sns.histplot(airbnb["minimum_nights"], bins=200)
plt.xlim(0, 100)
plt.title("Distribution of Minimum Nights (Zoomed)")
plt.show()
print("<--------------------------------------------------------------->")


print("<<<<< Group Comparison >>>>>")
sns.boxplot(x="room_type", y="price", data=airbnb)
plt.ylim(0, 1000)
plt.title("Price Distribution by Room Type")
plt.show() # แสดงให้เห็นความต่างของราคาในแต่ละหมวดเช่นห้องประเภทไหนค่าเฉลี่ยราคาสูง
sns.boxplot(x="neighbourhood_group", y="price", data=airbnb)
plt.ylim(0, 1000)
plt.title("Price by Neighbourhood Group")
plt.show() # เทียบราคาจากย่าน
print("<--------------------------------------------------------------->")


print("<<<<< Relationship Exploration >>>>>")
sns.scatterplot(x="minimum_nights", y="price", data=airbnb)
plt.xlim(0, 100)
plt.ylim(0, 2000)
plt.title("Price vs Minimum Nights")
plt.show() # หา relation ระหว่างราคากับจำนวนคืนขั้นต่ำแต่ที่ดูๆ เหมือนยิ่งจองระยะยาวราคาจะยิ่งถูกนะ
sns.scatterplot(x="number_of_reviews", y="price", data=airbnb)
plt.xlim(0, 200)
plt.ylim(0, 2000)
plt.title("Price vs Number of Reviews")
plt.show() # หา relation ระหว่างราคากับจำนวนรีวิว ยิ่งราคาสูงยิ่งมีรีวิวน้อยนะ อาจจะเป็นเพราะคนไม่ค่อยกล้ารีวิวที่พักที่แพงๆ555555
print("<--------------------------------------------------------------->")


print("<<<<< Summary Statistics of Price >>>>>") # Mean and Median โดยรวมของราคา อาจจะถูก distort เพราะมี data เสียแบบ คนตั้งราคาเป็น 0 นู่นนี่
print("Mean price:", airbnb["price"].mean())
print("Median price:", airbnb["price"].median())
print("Std price:", airbnb["price"].std())
print("IQR price:", airbnb["price"].quantile(0.75) - airbnb["price"].quantile(0.25))
print("<--------------------------------------------------------------->")


print("<<<<< Check For Outliers >>>>>")# Check outliers using IQR method, มีราคาที่อยู่นอก outliers อยู่ 2972 listing ซึ่งคิดเป็น 6% ซึ่งก็สมเหตุสมผลอยู่ถ้าเราดรอปพวก listing ที่แปลกๆเช่น ราคาเป็น 0 หรือ สูงปรี๊ดทั้งๆที่ไม่ใช่ luxury
Q1 = airbnb["price"].quantile(0.25)
Q3 = airbnb["price"].quantile(0.75)
IQR = Q3 - Q1
upper = Q3 + 1.5 * IQR
lower = Q1 - 1.5 * IQR
outliers = airbnb[(airbnb["price"] < lower) | (airbnb["price"] > upper)]
print("Number of price outliers:", outliers.shape[0])

print("<--------------------------------------------------------------->")
