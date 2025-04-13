library(ggplot2)
library(dplyr)
library(corrplot)
library(rpart)
library(caret)
library(ggparty)
library(torch)
library(smotefamily)

# Import dataset
white_wine_df = read.csv("/Users/siuuufuuu/Desktop/wine+quality/winequality-white.csv", sep = ";")
red_wine_df = read.csv("/Users/siuuufuuu/Desktop/wine+quality/winequality-red.csv", sep = ";")

#Inspect dataframe extracted
head(white_wine_df)
head(red_wine_df)

#Check if there are duplications 
sum(duplicated(red_wine_df))
sum(duplicated(white_wine_df))

#Check if there are na values
sum(is.na(red_wine_df))
sum(is.na(white_wine_df))

# Drop duplicate rows
red_wine_df <- unique(red_wine_df)
white_wine_df <- unique(white_wine_df)

# Check the summary for each dataframe
summary(red_wine_df)
summary(white_wine_df)

# Set up plotting area
par(mfrow = c(1, 3))

# Residual Sugar comparison
boxplot(red_wine_df$residual.sugar, white_wine_df$residual.sugar,
        main = "Residual Sugar",
        names = c("Red Wine", "White Wine"),
        col = c("#8B1A1A", "#FFFACD"))

# Free Sulfur Dioxide comparison
boxplot(red_wine_df$free.sulfur.dioxide, white_wine_df$free.sulfur.dioxide,
        main = "Free Sulfur Dioxide",
        names = c("Red Wine", "White Wine"),
        col = c("#8B1A1A", "#FFFACD"))

# Total Sulfur Dioxide comparison
boxplot(red_wine_df$total.sulfur.dioxide, white_wine_df$total.sulfur.dioxide,
        main = "Total Sulfur Dioxide",
        names = c("Red Wine", "White Wine"),
        col = c("#8B1A1A", "#FFFACD"))

# Add wine type column
red_wine_df$wine_type <- 0 # 0 refers to red wine
white_wine_df$wine_type <- 1 # 0 refers to white wine

# Create another numeric dataframe for SMOTE
wine_df_SMOTE <- rbind(red_wine_df, white_wine_df)

# Factorize data frame
red_wine_df$wine_type <- factor(red_wine_df$wine_type, levels = c(0, 1))
white_wine_df$wine_type <- factor(white_wine_df$wine_type, levels = c(0, 1))


# Merge using rbind()
wine_df <- rbind(red_wine_df, white_wine_df)

# Check data type
sapply(wine_df, class)
# As I will discretize data before modelling, we will leave variables as integer for visualization

table(wine_df$quality)
# Set up a plotting area with 4 rows and 4 columns
par(mfrow = c(4, 4))  

# Loop through each column and plot the histogram
for (col in colnames(wine_df)) {
  if (is.factor(wine_df[[col]])){
       barplot(table(wine_df[[col]]), 
       main = paste("Distribution of", col), 
       col = "#FF9999", 
       border = "white", 
       xlab = col)
  }
  else{
         hist(wine_df[[col]], 
         main = paste("Distribution of", col), 
         breaks = 20,
         col = "#FF9999", 
         border = "white", 
         xlab = col)
  }
}

# Reset plotting parameters
par(mfrow = c(1, 1))


# Plotting correaltion heatmap 
wine_df_wo_type <- wine_df %>% select(-wine_type)
wine_df_cor = cor(wine_df_wo_type)
corrplot(wine_df_cor, method = 'number')

# Create binned pH groups and calculate average quality
avg_quality_by_alcohol <- wine_df %>%
  mutate(alcohol_bin = cut(alcohol, breaks = 5)) %>%
  group_by(alcohol_bin, wine_type) %>%
  summarise(avg_quality = mean(quality, na.rm = TRUE))


# plotting the barplot
ggplot(avg_quality_by_alcohol, aes(x = alcohol_bin, y = avg_quality, fill = wine_type)) +
  geom_col(position = "dodge") +  
  labs(title = "Average Quality by alcohol Level and Wine Type",
       x = "Binned Alcohol level",
       y = "Average Quality",
       fill = "Wine Type" ) +
        scale_fill_manual(
         values = c("0" = "#FF9999", "1" = "#66B3FF"),  # Colors
         labels = c("Red Wine", "White Wine")  # Custom legend labels
       ) 


ggplot(wine_df, aes(x = factor(quality), y = alcohol, fill = quality)) +
  geom_boxplot(outlier.shape = NA) +
  labs(title = "Alcohol Content Distribution by Wine Quality",
       x = "Wine Quality",
       y = "Alcohol Content") +
  theme_minimal() +
  scale_fill_gradient(low = "darkred", high = "lightgreen")



# Calculate density for both red and white wines
x <- density(red_wine_df$quality)
y <- density(white_wine_df$quality)

# Plot the first density (red wine)
plot(y, main = "Density of Wine Quality (Red vs White)", 
     xlab = "Wine Quality", ylab = "Density", col = "darkblue", lwd = 2)

# Overlay the second density (white wine) on the same plot
lines(x, col = "#FF9999", lwd = 2)

# Add a legend
legend("topright", legend = c("White Wine", "Red Wine"), 
       col = c("darkblue", "#FF9999"), lwd = 2)









# Building CART model
# Separating response variable into 3 classes
wine_df$quality <- ifelse(
  wine_df$quality <= 5, "Low",ifelse(wine_df$quality == 6, "Med", "High"))
wine_df$wine_type <-ifelse(wine_df$wine_type == "0", "Red", "White")

table(wine_df$quality)
# Splitting test and train dataset
# As there are class imbalance, stratify is needed, 
# so we will use caret package for splitting.

# Set seed
set.seed(42)

# Selecting 80-20 split
samples <- createDataPartition(wine_df$quality, p = 0.8, list = FALSE)
training <- wine_df[samples,]
testing <-wine_df[-samples,]

# Training classification tree model
model_DT <- rpart(quality ~ .,data = training, method = "class", control = 
  rpart.control(minsplit = 100, cp = 0.001, maxdepth = 6))
printcp(model_DT)

# Prune model based on lowest xerror
pruned_model <- prune(model_DT, cp = 0.0047937)
plot(as.party(pruned_model))

# Check automatic feature selection
pruned_model$variable.importance

# Prediction
pred_CART <- predict(pruned_model, testing, type = "class")

# Construct a confusion matrix
confusion_matrix_CART <- confusionMatrix(pred_CART, as.factor(testing$quality))

# Common metrics 
accuracy_percentage <- confusion_matrix_CART$overall['Accuracy'] * 100
precision <- confusion_matrix_CART$byClass[,c("Sensitivity", "Specificity", "Pos Pred Value", "F1")]
colnames(precision) <- c("Recall", "Specificity", "Precision", "F1-Score")

# Print metrics
print(paste("Accuracy : ", accuracy_percentage))
print(precision)

# Plot the confusion matrix using ggplot2
ggplot(as.data.frame(confusion_matrix_CART$table), aes(x = Prediction, y = Reference,fill = Freq)) +
  geom_tile() + 
  scale_fill_gradient(low = "white", high = "darkred") +  # Color scale for frequency
  theme_minimal() +
  labs(title = "Confusion Matrix of Pruned model using CART", x = "Predicted", y = "Actual") +
  geom_text(aes(label = Freq), color = "black", size = 5)

# Get column mean for comparison
CART_metrics <- colMeans(precision)
CART_metrics['accuracy'] <- confusion_matrix_CART$overall['Accuracy']












# Building CART with oversampling (SMOTE)
# Cut the dataframe while maintaining numeric type
wine_df_SMOTE$quality <- ifelse(
  wine_df_SMOTE$quality <= 5 , 0,ifelse(wine_df_SMOTE$quality == 6, 1, 2))

# SMOTE
set.seed(42)
SMOTE_result <- SMOTE(X = wine_df_SMOTE, target = wine_df_SMOTE$quality,dup_size = 1.5)
wine_df_SMOTE_after <- SMOTE_result$data
table(wine_df_SMOTE_after$quality)
table(wine_df$quality)
wine_df_SMOTE_after <- wine_df_SMOTE_after %>%
  select(-quality) %>%  # Drop the old 'quality' column first
  rename(quality = class)


# Variable mapping
wine_df_SMOTE_after$quality <- ifelse(
  wine_df_SMOTE_after$quality == 0, "Low",ifelse(wine_df_SMOTE_after$quality == 1, "Med", "High"))
wine_df_SMOTE_after$wine_type <-ifelse(wine_df_SMOTE_after$wine_type == "0", "Red", "White")
summary(wine_df_SMOTE_after)


# Selecting 80-20 split
samples_SMOTE <- createDataPartition(wine_df_SMOTE_after$quality, p = 0.8, list = FALSE)
training_SMOTE <- wine_df_SMOTE_after[samples_SMOTE,]
testing_SMOTE <-wine_df_SMOTE_after[-samples_SMOTE,]

# Training classification tree model
model_DT_SMOTE <- rpart(quality ~ .,data = training_SMOTE, method = "class", control = 
                    rpart.control(minsplit = 100, cp = 0.0005, maxdepth = 6))
printcp(model_DT_SMOTE)


# Prune model based on lowest xerror
pruned_model_SMOTE <- prune(model_DT_SMOTE, cp = 0.00343107)

plot(as.party(pruned_model_SMOTE))

# Check automatic feature selection
pruned_model_SMOTE$variable.importance

# Prediction
pred_CART_SMOTE <- predict(pruned_model_SMOTE, testing_SMOTE, type = "class")

# Construct a confusion matrix
confusion_matrix_CART_SMOTE <- confusionMatrix(pred_CART_SMOTE, as.factor(testing_SMOTE$quality))

# Common metrics 
accuracy_percentage_SMOTE <- confusion_matrix_CART_SMOTE$overall['Accuracy'] * 100
precision_SMOTE <- confusion_matrix_CART_SMOTE$byClass[,c("Sensitivity", "Specificity", "Pos Pred Value", "F1")]
colnames(precision_SMOTE) <- c("Recall", "Specificity", "Precision", "F1-Score")

# Print metrics
print(paste("Accuracy : ", accuracy_percentage_SMOTE))
print(precision_SMOTE)

# Plot the confusion matrix using ggplot2
ggplot(as.data.frame(confusion_matrix_CART_SMOTE$table), aes(x = Prediction, y = Reference,fill = Freq)) +
  geom_tile() + 
  scale_fill_gradient(low = "white", high = "darkred") +  # Color scale for frequency
  theme_minimal() +
  labs(title = "Confusion Matrix of Pruned model using CART with SMOTE", x = "Predicted", y = "Actual") +
  geom_text(aes(label = Freq), color = "black", size = 5)

# Get column mean for comparison
CART_metrics_SMOTE <- colMeans(precision_SMOTE)
CART_metrics_SMOTE['accuracy'] <- confusion_matrix_CART_SMOTE$overall['Accuracy']








# Building up FNN network

# Convert data frame to numerical values
# Reuse Numerical dataframe created for SMOTE
head(wine_df_SMOTE)

#Rename and check dtype as torch only support numeric and boolean
sapply(wine_df_SMOTE,class)
wine_df_NN <- wine_df_SMOTE

# Min-Max scaling as it have defined range
preProc <- preProcess(wine_df_NN[, -which(names(wine_df_NN) == "quality")], method = "range")
wine_df_NN_scaled <- predict(preProc, wine_df_NN[, -which(names(wine_df_NN) == "quality")])
# Combine dataset
wine_df_NN_scaled <- cbind(wine_df_NN_scaled, quality = wine_df_NN$quality)

#Splitting into train and test set
Train_NN <- wine_df_NN_scaled[samples,]
Test_NN <- wine_df_NN_scaled[-samples,]

# Separate features and labels for training
X_train_NN <- Train_NN[, -which(names(Train_NN) == "quality")]
y_train_NN <- Train_NN[, which(names(Train_NN) == "quality")]

# Separate features and labels for testing
X_test_NN <- Test_NN[, -which(names(Test_NN) == "quality")]
y_test_NN <- Test_NN[, which(names(Test_NN) == "quality")]

# Change data type for one-hot encoding
y_train_NN <- factor(y_train_NN)

# One hot encoding for target variable as its a multi-class classification
dummy_model <- dummyVars(formula = ~ .-1, data = data.frame(quality = y_train_NN), fullRank = FALSE)
y_train_NN_encoded <- predict(dummy_model, newdata = data.frame(quality = y_train_NN))


# Add +1 after one-hot encoding as R indexing starts at 1
y_train_NN_encoded <- y_train_NN_encoded + 1


# Converting to data into tensor for training
X_train_temp <- as.matrix(X_train_NN)
X_test_temp <-as.matrix(X_test_NN)
X_train_tensor <- torch_tensor(X_train_temp, dtype = torch_float())
X_test_tensor <-torch_tensor(X_test_temp, dtype = torch_float())
y_train_tensor <- torch_tensor(y_train_NN_encoded, dtype = torch_float())



# Setup the model
torch_manual_seed(42)
nn_model = nn_sequential(
  nn_linear(12,24), # Input layer (12-32 neurons)
  nn_relu(), # Activation function
  nn_linear(24,12), # Input layer (12-32 neurons)
  nn_relu(), # Activation function
  nn_linear(12,3) # Output layer (16 - 3 neurons)
) 
# As i am using Adam loss function, torch will apply softmax in output layer

nn_model$parameters
loss_fn <- nn_cross_entropy_loss()
optimizer <- optim_adam(nn_model$parameters, lr = 0.05)
epochs <- 5000
batch_size <- 128

for (epoch in 1:epochs) {
  nn_model$train()  # Set the model to training mode
  optimizer$zero_grad()  # Clear the gradients
  
  # Forward pass
  predictions <- nn_model(X_train_tensor)
  
  # Compute loss
  loss <- loss_fn(predictions, y_train_tensor)
  
  # Calculate accuracy
  # Get the predicted class labels (indices)
  predicted_classes <- predictions$argmax(dim = 2)
  
  # Compare predicted class labels with true labels (y_train_tensor)
  #correct_preds <- torch_eq(predicted_classes, y_train_tensor)
  pred_int <- as.integer(predicted_classes)
  y_train_NN_int <- as.integer(y_train_NN)
  # Calculate accuracy
  accuracy <- sum(pred_int == y_train_NN_int) / length(y_train_NN_int) # Fraction of correct predictions
  
  # Backward pass
  loss$backward()
  
  # Update weights
  optimizer$step()
  
  # Print loss for every 10th epoch
  if (epoch %% 10 == 0) {
    cat("Epoch: ", epoch, " Loss: ", loss$item(), " Accuracy: ", accuracy, "\n")
  }
}

nn_model$eval()  # Set the model to evaluation mode
output <- nn_model(X_test_tensor)
predicted_classes <- torch_max(output, dim = 2)[[2]]
predicted_r <- as.integer(predicted_classes) - 1 
accuracy <- sum(predicted_r == (y_test_NN)) / length(y_test_NN)
print(paste("Accuracy: ", accuracy))

# Construct a confusion matrix
confusion_matrix_NN <- confusionMatrix(as.factor(predicted_r), as.factor(y_test_NN))
confusion_matrix_NN
# Common metrics 
accuracy_percentage <- confusion_matrix_NN$overall['Accuracy'] * 100
precision <- confusion_matrix_NN$byClass[,c("Sensitivity", "Specificity", "Pos Pred Value", "F1")]
colnames(precision) <- c("Recall", "Specificity", "Precision", "F1-Score")

# Print metrics
print(paste("Accuracy : ", accuracy_percentage))
print(precision)

# Define class labels 
class_labels <- c("0" = "High", "1" = "Medium", "2" = "Low")

# Plot the confusion matrix using ggplot2
ggplot(as.data.frame(confusion_matrix_NN$table), aes(x = Prediction, y = Reference,fill = Freq)) +
  geom_tile() + 
  scale_fill_gradient(low = "white", high = "darkred") +  # Color scale for frequency
  labs(title = "Confusion Matrix of FNN model as Multi-class classification", x = "Predicted", y = "Actual") +
  geom_text(aes(label = Freq), color = "black", size = 5) +
  scale_x_discrete(labels = class_labels) +  
  scale_y_discrete(labels = class_labels)

# Get column mean for comparison
NN_metrics <- colMeans(precision)
NN_metrics['accuracy'] <- confusion_matrix_NN$overall['Accuracy']
NN_metrics
















# Building up FNN network (with SMOTE)

# Convert data frame to numerical values
# Reuse Numerical dataframe created for SMOTE
wine_df_NN_SMOTE <- SMOTE_result$data

wine_df_NN_SMOTE <- wine_df_NN_SMOTE %>%
  select(-quality) %>%  # Drop the old 'quality' column first
  rename(quality = class)

#Rename and check dtype as torch only support numeric and boolean
sapply(wine_df_NN_SMOTE,class)
head(wine_df_NN_SMOTE)
wine_df_NN_SMOTE$quality <- as.numeric(wine_df_NN_SMOTE$quality)

# Min-Max scaling as it have defined range
preProc <- preProcess(wine_df_NN_SMOTE[, -which(names(wine_df_NN_SMOTE) == "quality")], method = "range")
wine_df_NN_scaled <- predict(preProc, wine_df_NN_SMOTE[, -which(names(wine_df_NN_SMOTE) == "quality")])
# Combine dataset
wine_df_NN_scaled <- cbind(wine_df_NN_scaled, quality = wine_df_NN_SMOTE$quality)

#Splitting into train and test set
Train_NN <- wine_df_NN_scaled[samples_SMOTE,]
Test_NN <- wine_df_NN_scaled[-samples_SMOTE,]

# Separate features and labels for training
X_train_NN <- Train_NN[, -which(names(Train_NN) == "quality")]
y_train_NN <- Train_NN[, which(names(Train_NN) == "quality")]

# Separate features and labels for testing
X_test_NN <- Test_NN[, -which(names(Test_NN) == "quality")]
y_test_NN <- Test_NN[, which(names(Test_NN) == "quality")]

# Change data type for one-hot encoding
y_train_NN <- factor(y_train_NN)

# One hot encoding for target variable as its a multi-class classification
dummy_model <- dummyVars(formula = ~ .-1, data = data.frame(quality = y_train_NN), fullRank = FALSE)
y_train_NN_encoded <- predict(dummy_model, newdata = data.frame(quality = y_train_NN))


# Add +1 after one-hot encoding as R indexing starts at 1
y_train_NN_encoded <- y_train_NN_encoded + 1


# Converting to data into tensor for training
X_train_temp <- as.matrix(X_train_NN)
X_test_temp <-as.matrix(X_test_NN)
X_train_tensor <- torch_tensor(X_train_temp, dtype = torch_float())
X_test_tensor <-torch_tensor(X_test_temp, dtype = torch_float())
y_train_tensor <- torch_tensor(y_train_NN_encoded, dtype = torch_float())




# Setup the model
torch_manual_seed(42)
nn_model = nn_sequential(
  nn_linear(12,36), # Input layer (12-32 neurons)
  nn_relu(), # Activation function
  nn_linear(36, 12), # Dense layer(32-16 neurons)
  nn_relu(),
  nn_linear(12,3) # Output layer (16 - 3 neurons)
) 
# As i am using Adam loss function, torch will apply softmax in output layer

nn_model$parameters
loss_fn <- nn_cross_entropy_loss()
optimizer <- optim_adam(nn_model$parameters, lr = 0.05)
epochs <- 5000
batch_size <- 128

for (epoch in 1:epochs) {
  nn_model$train()  # Set the model to training mode
  optimizer$zero_grad()  # Clear the gradients
  
  # Forward pass
  predictions <- nn_model(X_train_tensor)
  
  # Compute loss
  loss <- loss_fn(predictions, y_train_tensor)
  
  # Calculate accuracy
  # Get the predicted class labels (indices)
   predicted_classes <- predictions$argmax(dim = 2)

  # Compare predicted class labels with true labels (y_train_tensor)
  #correct_preds <- torch_eq(predicted_classes, y_train_tensor)
  pred_int <- as.integer(predicted_classes)
  y_train_NN_int <- as.integer(y_train_NN)
  # Calculate accuracy
  accuracy <- sum(pred_int == y_train_NN_int) / length(y_train_NN_int) # Fraction of correct predictions
  
  # Backward pass
  loss$backward()
  
  # Update weights
  optimizer$step()
  
  # Print loss for every 10th epoch
  if (epoch %% 10 == 0) {
    cat("Epoch: ", epoch, " Loss: ", loss$item(), " Accuracy: ", accuracy, "\n")
  }
}

nn_model$eval()  # Set the model to evaluation mode
output <- nn_model(X_test_tensor)
predicted_classes <- torch_max(output, dim = 2)[[2]]
predicted_r <- as.integer(predicted_classes) - 1 
accuracy <- sum(predicted_r == (y_test_NN)) / length(y_test_NN)
print(paste("Accuracy: ", accuracy))

# Construct a confusion matrix
confusion_matrix_NN <- confusionMatrix(as.factor(predicted_r), as.factor(y_test_NN))
confusion_matrix_NN
# Common metrics 
accuracy_percentage <- confusion_matrix_NN$overall['Accuracy'] * 100
precision <- confusion_matrix_NN$byClass[,c("Sensitivity", "Specificity", "Pos Pred Value", "F1")]
colnames(precision) <- c("Recall", "Specificity", "Precision", "F1-Score")

# Print metrics
print(paste("Accuracy : ", accuracy_percentage))
print(precision)

# Define class labels 
class_labels <- c("0" = "High", "1" = "Medium", "2" = "Low")

# Plot the confusion matrix using ggplot2
ggplot(as.data.frame(confusion_matrix_NN$table), aes(x = Prediction, y = Reference,fill = Freq)) +
  geom_tile() + 
  scale_fill_gradient(low = "white", high = "darkred") +  # Color scale for frequency
  labs(title = "Confusion Matrix of FNN model as Multi-class classification with SMOTE", x = "Predicted", y = "Actual") +
  geom_text(aes(label = Freq), color = "black", size = 5) +
  scale_x_discrete(labels = class_labels) +  
  scale_y_discrete(labels = class_labels)

# Get column mean for comparison
NN_metrics_SMOTE <- colMeans(precision)
NN_metrics_SMOTE['accuracy'] <- confusion_matrix_NN$overall['Accuracy']
NN_metrics








# Addition : using adas to oversample
# Building CART with oversampling (SMOTE)
# Cut the dataframe while maintaining numeric type



wine_df_SMOTE <- rbind(red_wine_df, white_wine_df)
wine_df_SMOTE$wine_type <- as.numeric(wine_df_SMOTE$wine_type)
wine_df_SMOTE$quality <- ifelse(
  wine_df_SMOTE$quality <= 5 , 0,ifelse(wine_df_SMOTE$quality == 6, 1, 2))

# SMOTE
set.seed(42)
ADAS_result <- ADAS(X = wine_df_SMOTE, target = wine_df_SMOTE$quality)
sapply(wine_df_SMOTE,class)
wine_df_ADAS_after <- ADAS_result$data
table(wine_df_ADAS_after$quality)
table(wine_df$quality)
wine_df_ADAS_after <- wine_df_ADAS_after %>%
  select(-quality) %>%  # Drop the old 'quality' column first
  rename(quality = class)


# Variable mapping
wine_df_ADAS_after$quality <- ifelse(
  wine_df_ADAS_after$quality == 0, "Low",ifelse(wine_df_ADAS_after$quality == 1, "Med", "High"))
wine_df_ADAS_after$wine_type <-ifelse(wine_df_ADAS_after$wine_type == "0", "Red", "White")
summary(wine_df_ADAS_after)


# Selecting 80-20 split
samples_ADAS <- createDataPartition(wine_df_ADAS_after$quality, p = 0.8, list = FALSE)
training_ADAS <- wine_df_ADAS_after[samples_ADAS,]
testing_ADAS <-wine_df_ADAS_after[-samples_ADAS,]

# Training classification tree model
model_DT_ADAS <- rpart(quality ~ .,data = training_ADAS, method = "class", control = 
                          rpart.control(minsplit = 100, cp = 0.0005, maxdepth = 6))
printcp(model_DT_ADAS)


# Prune model based on lowest xerror
pruned_model_ADAS <- prune(model_DT_ADAS, cp = 0.00086957)

plot(as.party(pruned_model_ADAS))

# Check automatic feature selection
pruned_model_ADAS$variable.importance

# Prediction
pred_CART_ADAS <- predict(pruned_model_ADAS, testing_ADAS, type = "class")

# Construct a confusion matrix
confusion_matrix_CART_ADAS <- confusionMatrix(pred_CART_ADAS, as.factor(testing_ADAS$quality))

# Common metrics 
accuracy_percentage_ADAS <- confusion_matrix_CART_ADAS$overall['Accuracy'] * 100
precision_ADAS <- confusion_matrix_CART_ADAS$byClass[,c("Sensitivity", "Specificity", "Pos Pred Value", "F1")]
colnames(precision_ADAS) <- c("Recall", "Specificity", "Precision", "F1-Score")

# Print metrics
print(paste("Accuracy : ", accuracy_percentage_ADAS))
print(precision_ADAS)

# Plot the confusion matrix using ggplot2
ggplot(as.data.frame(confusion_matrix_CART_ADAS$table), aes(x = Prediction, y = Reference,fill = Freq)) +
  geom_tile() + 
  scale_fill_gradient(low = "white", high = "darkred") +  # Color scale for frequency
  theme_minimal() +
  labs(title = "Confusion Matrix of Pruned model using CART with SMOTE", x = "Predicted", y = "Actual") +
  geom_text(aes(label = Freq), color = "black", size = 5)


cbind(CART_metrics,CART_metrics_SMOTE,NN_metrics,NN_metrics_SMOTE)
CART_metrics_SMOTE
NN_metrics
NN_metrics_SMOTE
