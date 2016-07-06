#Movie Box Office Predictor

![](https://raw.githubusercontent.com/Hongtian22/capstone_project/master/Pics/hollywood-box-office-collection-earning-records-hit-flop-movies-list-231181951%20copy.jpg)

## Purpose
The purpose of my project is to predict the second week box office number of a movie, based on its basic information, including first week box office, production budget, actors, directors, etc.


## Data

My data came from two website. IMDb.com provides movies' basic information. BoxOfficeMojo.com provides detailed box office information.


## Features
One important thing about my project is how to transform real world informatin into useful numeric features for my models. 
- Inflation
	Inflation factors is very important when consider price over time.
- Regularization
	Some numbers are huge, like 



## Models and Results
Finally, I built several multiple linear regression models with ridge regularization. I grouped movies by different genres, different budget range, different 'Movie stars' range. The average accuracy cross all the models are about 82%.


## In the Future

In the future, I want to introduce movie awards information and rating information in my model, to predict the box office in the following weeks.






