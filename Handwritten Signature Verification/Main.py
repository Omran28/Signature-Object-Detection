from Loss_Function import *
from Train_Test_Results import *
from tensorflow.keras.optimizers import Adam


# Creating triplets
train_Triplets = Creating_Triplets(1)
test_Triplets = Creating_Triplets(0)

# Loading external model
input_shape = (128, 128, 3)
siamese_network = get_siamese_network(input_shape)
siamese_network.summary()

# Creating loss function
siamese_model = SiameseModel(siamese_network)
optimizer = Adam(learning_rate=1e-3, epsilon=1e-01)
siamese_model.compile(optimizer=optimizer)

# Training
siamese_model = Train(train_Triplets, test_Triplets, siamese_model)

# Extract encoder
encoder = Extract_encoder(siamese_model)
encoder.save_weights("encoder")
encoder.summary()

# Classify test data
Classification(test_Triplets, encoder)
