from model import model, scaler

# Make a prediction
new_value = [[0.19499106, 0.32795699, 0.38976378, 1, 0, 1, 0, 0,
              0]]  # enter new data in 2D array. Only numbers + dummy variables.
new_value = scaler.transform(new_value)  # Don't forget to scale!
print(model.predict(new_value))