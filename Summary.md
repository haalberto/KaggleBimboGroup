#Summary of project: Grupo Bimbo Inventory Demand

## Introduction

This repository contains the scripts I used in my final submission to the Kaggle competition named Bimbo Group Inventory Demand. In this competition we are asked to predict the demand of products of Grupo Bimbo, a company in Mexico which produces bakery goods. Specifically, we are provided with records indicating the sales of Bimbo products at different stores for a duration of 7 consecutive weeks and asked to predict the demand of these same products for each store on weeks 8 and 9. An important point to note is that on the last two weeks there are costumers and products that were not present before, so the model needs to be able to handle stores and products for which there are no records in the training data.

## Data set

The main file, train.csv, contains roughly 70 million records with the following columns:

Semana — Week number

Agencia_ID — Sales depot ID

Canal_ID — Sales channel ID

Ruta_SAK — Route ID (Several routes form a sales depot)

Cliente_ID — Client ID (corresponds to a store that sells Bimbo products)

NombreCliente — Client name

Producto_ID — Product ID

NombreProducto — Product Name

Venta_uni_hoy — Sales unit this week (integer)

Venta_hoy — Sales this week (unit: pesos)

Dev_uni_proxima — Returns unit next week (integer)

Dev_proxima — Returns next week (unit: pesos)

Demanda_uni_equil — Adjusted Demand (integer) (This is the target I need to predict)

We are asked to use this information to train a model and predict the demand for a test set provided by Kaggle. The test set only contains the columns corresponding to week number and to the ID’s of the sales depot, channel, route, client, and product.

In addition to that data, we are given three spreadsheets with the name of each client, the description of each product, and the towns and states that each sales depot services. These spreadsheets are joinable to the main file using the appropriate ID.

## Evaluation metric

The evaluation metric for this competition is the Root Mean Squared Logarithmic Error (RMSLE), where the logarithmic error is defined as
LE = log (p+1) – log(a+1),
where p is the predicted value of demand and a is the actual value.

## Feature engineering

Most of the features I created are based on group averages of the log demand, where the log demand is defined as LD = log (p+1). I used the log demand instead of the regular demand because it is more in line with the evaluation metric. My reasoning for using group averages was the following: The most basic model consists of predicting the same value for all products are all stores, and the best choice of this value is the average of the log demand of all the training records, since our target is the RMSLE. Thus, I decided to improve on this idea by calculating averages by group (which is a step up from simply calculating the average for all records) and using those as features for machine learning algorithms.

The features I created were the following:
Cliente_mean: Group average of the log demand of the records for a given store, regardless of what those products are. This provides a metric of the volume of sales for that store.
C_unique: Number of different products that a store has carried. Measures the variety of products that the store handles and is complementary to the above.
Producto_mean: Group average of the log demand of the records for a product, without distinguishing between stores. This provides a metric of how popular this product is overall.
P_unique: Number of different stores that have carried a given product. Measures how widespread a product is and is complementary to the above.
CP_mean: Group average for a given store-product combination. This potentially has great predictive power when both the store and product are known, but not useful when handling a new store or a new product.
PR_mean: Group average for a given product-route combination. Very useful when the store is new since at least we know what route it belongs to.

I calculated each of the above quantities and joined them to the sales records using the corresponding ID’s. 

##Models

The models I trained used linear regression to predict the log demand. I made this choice because then the minimization objective of the regression corresponds exactly to the RMSLE. An idea for future improvement is to try nonlinear models which use the same or at least a similar minimization objective. 

I trained four different models: One using all the features and three that use only use a subset of features. The models were the following:
1. Full model: Uses all features.
2. Client-only model: Only uses the features related to the store (does not use product information)
3. Product-only model: Only uses features related to the product (and route), but does not use store information.
4. No C-P model: As the full model, minus the CP_mean feature. This model is useful when both the client and product are known, but combination has not occurred in any record before.


The reason for training these four models is that in the Kaggle test set there are records with products and stores for which there is no data in the previous weeks. Thus, some of the features cannot apply to the test set. For example, if a product is new there is no average over previous records of that product. Thus, we cannot apply the full model to records with new products, but we could use the client-only model to make an educated guess based on the volume of sales of that store and the number of different products it carries.

Thus, in making predictions for the Kaggle test set, I used the full model for records were all the features were applicable. If both store and product were known but had not occurred in combination, I used the No-CP model. If the client was known but the product was new or vice versa, I used the client-only or product-only models. Finally, in the worst case when neither the costumer nor the product were known, I made a guess by using the average of the log demand of all records.

The training of the linear regression models was done by holding out data from the first week (actually called week 3 since the numbering went from 3 to 10 instead of 1 to 8) to use for validation. This validation set was used to choose the value of the regularization parameter.

## Description of the scripts

reg_averages_ext4_poly.py: Trains the full model and saves a pickled object containing the model (full_deg1.pkl)
reg_seg_nocp.py: Trains the No C-P model and saves it (semi_full.pkl)
reg_seg_nocust.py: Trains the Product-only model and saves it (no_costumer_deg1.pkl)
reg_seg_noprod.py: Trains the Client-only model and saves it (no_product_deg1.pkl)
reg_seg_naive.py: Calculates the average of the log demand over all records and saves it (fixed_value.pkl)
reg_seg_submit.py: Uses all the trained models to predict the values of the test set and creates a file for submission to Kaggle.
