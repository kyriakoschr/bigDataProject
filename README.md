# bigDataProject

Title: Restaurants classification using Apache Spark
# To do(demetris)

- [x] transform existing code to a function
- [x] make a main and call them from there
- [x] transform ratings 1,2,3->0 4,5->1

# To do(kyriakos)
- [ ] make a function for each classifier 1/3
- [ ] make a function to call them

#Function Main()
-  if (file == train_set)
    - [ ] parse the file and remove stopwords
    - [ ] replace [1, 2, 3] to 0 and [4, 5] to 1
    - [ ] classify with Logistic Regression
    - [ ] classify with Navy Bayes
    - [ ] classify with Random Forest
    - [ ] compare the result and select the most accuracy classifier
-  if (file == test_set)
    - [ ] parse the file and remove stopwords
    - [ ] classify with the best accuracy classifier
    - [ ] compare the results with the train_set