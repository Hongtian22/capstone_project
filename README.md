#Movie Box Office Predictor

![](https://raw.githubusercontent.com/Hongtian22/capstone_project/master/Pics/hollywood-box-office-collection-earning-records-hit-flop-movies-list-231181951%20copy.jpg)

## Purpose
The purpose of my project is to predict the second week box office number of a movie, based on its basic information, including first week box office, production budget, actors, directors, etc.


## Data

My data came from two website. IMDb.com provides movies' basic information. BoxOfficeMojo.com provides detailed box office information.


## Features
One important thing about my project is how to transform real world informatin into useful numeric features for my models. 
- Inflation: Inflation factors is very important when consider price over time. I used the CPI numbers came from Bureau of Labor Statistics.
- Regularization: Some numbers are huge, like million dollars box office. Some numbers are small, like 10 theaters this movie was on. I used Ridge regularization to solve this problem.
- Other movies:  Say, if you are the only movie in that week, you will get all the attention you want. But if you are in the same week with Titanic, well, good luck. So in my model, I calculated the total box office number for that week, in the whole US movie market. Then for each movie, got the ratio of week 1 box office over the total box office. That ratio will represent the market share this movie got in that week.
- Actors: 


## Models and Results
Finally, I built several multiple linear regression models with ridge regularization. I grouped movies by different genres, different budget range, different 'Movie stars' range. The average accuracy cross all the models are about 82%.


## In the Future

In the future, I want to introduce movie awards information and rating information in my model, to predict the box office in the following weeks.






