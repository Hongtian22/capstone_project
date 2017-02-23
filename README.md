#Movie Box Office Predictor

![](https://raw.githubusercontent.com/Hongtian22/capstone_project/master/Pics/hollywood-box-office-collection-earning-records-hit-flop-movies-list-231181951%20copy.jpg)

## Purpose
The purpose of my project is to predict the second week box office number of a movie, based on its basic information, including first week box office, production budget, actors, directors, etc.


## Data

My data came from two website.    
- IMDb.com provides movies' basic information.    
- BoxOfficeMojo.com provides detailed box office information.


## Features
One important thing about my project is how to transform real world informatin into useful numeric features for my models. 
- Inflation: Inflation factors is very important when consider price over time. I used the CPI numbers came from Bureau of Labor Statistics.
- Regularization: Some numbers are huge, like million dollars box office. Some numbers are small, like 10 theaters this movie was on. I used Ridge regularization to solve this problem.
- Other movies:  Say, if you are the only movie in that week, you will get all the attention you want. But if you are in the same week with Titanic, well, good luck. So in my model, I calculated the total box office number for that week, in the whole US movie market. Then for each movie, got the ratio of week 1 box office over the total box office. That ratio will represent the market share this movie got in that week.
- Actors: Are these two guys the same person?   

![](https://raw.githubusercontent.com/Hongtian22/capstone_project/master/Pics/MV5BMjE1NzI2NzgzMV5BMl5BanBnXkFtZTcwNTAwOTYwMw%40%40._V1__SX1394_SY749_.jpg)
![](https://raw.githubusercontent.com/Hongtian22/capstone_project/master/Pics/revenant%E2%80%93il-redivivo.jpg)    

Well, they share the same name. However, their contribution to their movies are totally different, and what people expect from them are totally different. So I can't juse use the names as a feature in my models, not mention there are way too many. Instead, I calculated cumulated box office for each actor, from the very beginning of his/her history until the release date for that single movie. Then add up the most 4 important actors / actress in that movie. This number perfectly reflects the actors factor. 


## Models and Results
Finally, I built several multiple linear regression models with ridge regularization. I grouped movies by different genres, different budget range, different 'Movie stars' range. The average accuracy cross all the models are about 82%.      
![](https://raw.githubusercontent.com/Hongtian22/capstone_project/master/Pics/movie.jpg)
![](https://raw.githubusercontent.com/Hongtian22/capstone_project/master/Pics/movie1.jpg)
![](https://raw.githubusercontent.com/Hongtian22/capstone_project/master/Pics/movie2.jpg)

## In the Future

In the future, I want to introduce movie awards information and rating information in my model, to predict the box office in the following weeks.   
Most awards and ratings came several weeks, sometimes even months, after the release date. So I can't include these factors in my current models. When I build models for the following weeks, these factors will be useful.






