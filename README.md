# 📊 BigMart Sales Prediction  

## 🔍 Project Overview  
This project aims to **predict sales for BigMart outlets** using historical sales data from 2013. The dataset includes information on **1559 products** across **10 stores** in different cities. Using this predictive model, BigMart can identify key factors that impact sales and improve its inventory and marketing strategies.  

Since some stores might not have complete data due to technical issues, we performed **data cleaning and preprocessing** to handle missing values.  

---

## 📂 Data Description  

### **1. Train Dataset (8523 records)**  
This dataset contains past sales data, including both input features and the target variable `Item_Outlet_Sales`, which we aim to predict.  

| **Variable**                 | **Description** |
|------------------------------|----------------|
| **Item_Identifier**          | Unique product ID |
| **Item_Weight**              | Weight of the product |
| **Item_Fat_Content**         | Whether the product is low-fat or regular |
| **Item_Visibility**          | The percentage of total display area allocated to the product |
| **Item_Type**                | Category of the product |
| **Item_MRP**                 | Maximum Retail Price of the product |
| **Outlet_Identifier**        | Unique store ID |
| **Outlet_Establishment_Year** | Year the store was established |
| **Outlet_Size**              | Size of the store (Small, Medium, Large) |
| **Outlet_Location_Type**     | Type of city (Tier 1, Tier 2, Tier 3) |
| **Outlet_Type**              | Type of store (Grocery store, Supermarket Type 1, 2, or 3) |
| **Item_Outlet_Sales**        | Sales of the product at the particular store (Target Variable) |  

---

### **2. Test Dataset (5681 records)**  
This dataset contains all the input variables except for `Item_Outlet_Sales`, which needs to be predicted.  

| **Variable**                 | **Description** |
|------------------------------|----------------|
| **Item_Identifier**          | Unique product ID |
| **Item_Weight**              | Weight of the product |
| **Item_Fat_Content**         | Whether the product is low-fat or regular |
| **Item_Visibility**          | The percentage of total display area allocated to the product |
| **Item_Type**                | Category of the product |
| **Item_MRP**                 | Maximum Retail Price of the product |
| **Outlet_Identifier**        | Unique store ID |
| **Outlet_Establishment_Year** | Year the store was established |
| **Outlet_Size**              | Size of the store (Small, Medium, Large) |
| **Outlet_Location_Type**     | Type of city (Tier 1, Tier 2, Tier 3) |
| **Outlet_Type**              | Type of store (Grocery store, Supermarket Type 1, 2, or 3) |  

---

## 🔧 Data Preprocessing  

### **1. Handling Missing Values**  
- **`Item_Weight`**: Filled missing values using the average weight of similar products.  
- **`Outlet_Size`**: Imputed missing values based on store type and location.  

### **2. Standardizing Data Formats**  
- Corrected inconsistencies in `Item_Fat_Content` (merged "low fat" and "LF" into "Low Fat").  
- Converted `Outlet_Establishment_Year` to store age using:  
  ```python
  Store_Age = 2024 - Outlet_Establishment_Year
