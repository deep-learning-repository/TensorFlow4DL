import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

bank_df = pd.read_csv('bank-additional-full.csv', sep=';')
# binary encoding of class label
bank_df['y'] = bank_df['y'].map({'no': 0, 'yes': 1})
# categorical features
categorical = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
# Perform feature scaling using MinMaxScaler
# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler()  # default= (0, 1)
numerical = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
             'euribor3m', 'nr.employed']
bank_df[numerical] = scaler.fit_transform(bank_df[numerical])
bank_df = pd.get_dummies(bank_df)  # encods categorical data
bank_df = bank_df.drop('duration', axis=1)
bank_df_y = bank_df['y'].values.reshape(-1, 1)
train_set, test_set, train_set_y, test_set_y = train_test_split(bank_df, bank_df_y, test_size=0.2, random_state=0)
# make sure negative class is distributed equally in train and test set.
print("Distribution of negative calss:\n", train_set['y'].value_counts() / len(train_set))

# Perform upsampling to address sample imbalance
# Separate majority and minority classes
from sklearn.utils import resample

train_negative = train_set[train_set['y'] == 0]
train_positive = train_set[train_set['y'] == 1]

# Upsample minority class
train_positive_upsample = resample(train_positive,
                                   replace=True,  # sample with replacement
                                   n_samples=29238,  # to match majority class
                                   random_state=18  # reproducible results
                                   )
# Combine majority class with upsampled minority class
train_upsample = pd.concat([train_negative, train_positive_upsample])

# Display new class counts
print("Display new class counts:\n", train_upsample['y'].value_counts())

# Create X, y for upsampled training and testing
X_train = train_upsample.drop('y', axis=1)
X_test = test_set.drop('y', axis=1)
y_train = train_upsample['y']
y_test = test_set['y']
# create X, y for imbalanced test set for performance validation
X_imb = test_set.drop('y', axis=1)
y_imb = test_set['y']
y_train = y_train.values.reshape(-1, 1)
X_train, X_valid, y_train, y_valid = train_test_split(X_train.values,
                                                      y_train,
                                                      test_size=0.2,
                                                      random_state=18)


n_features = X_train.shape[1]
m_train = X_train.shape[0]
n_output = 1
n_hidden = 16
learning_rate = 0.001
n_epochs = 20
batch_size = 128
droup_out = 0.4

tf.reset_default_graph()
with tf.name_scope('placeholders'):
    X = tf.placeholder(shape=(None, n_features), dtype=tf.float32)
    y = tf.placeholder(shape=(None, n_output), dtype=tf.float32)
    keep_prob = tf.placeholder(tf.float32)

with tf.name_scope('hidden_layer'):
    W = tf.Variable(tf.random_normal((n_features, n_hidden)))
    b = tf.Variable(tf.zeros((n_hidden,)))
    X_hidden = tf.nn.relu(tf.matmul(X, W) + b)
    # Apply dropout
    X_hidden = tf.nn.dropout(X_hidden, keep_prob=keep_prob)

with tf.name_scope("output"):
    W = tf.Variable(tf.random_normal((n_hidden, n_output)))
    b = tf.Variable(tf.zeros((n_output,)))
    y_logit = tf.matmul(X_hidden, W) + b
    # the sigmoid gives the class probability of 1
    y_one_prob = tf.sigmoid(y_logit)
    # Rounding p(y=1) will give the correct prediction.
    y_pred = tf.round(y_one_prob)

with tf.name_scope("loss"):
    # Compute the cross-entropy term for each datapoint
    entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=y)
    # Sum all contributions
    l = tf.reduce_sum(entropy)

with tf.name_scope("summaries"):
    tf.summary.scalar("loss", l)
    merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter('./nn_train', tf.get_default_graph())

with tf.name_scope("optim"):
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(l)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    for epoch in range(1, n_epochs + 1):
        batch_pos = 0
        pos = 0
        while pos < m_train:
            X_batch = X_train[pos: pos + batch_size]
            y_batch = y_train[pos: pos + batch_size]
            feed_dict = {X: X_batch, y: y_batch, keep_prob: 0.5}
            _, summary, loss = sess.run([train_op, merged, l], feed_dict=feed_dict)
            print(f"epoch {epoch}, step {step}, loss: {loss}")
            train_writer.add_summary(summary, step)

            step += 1
            pos += batch_size
    # Make Predictions 
    y_train_pred = sess.run(y_pred, feed_dict={X: X_train, keep_prob: 1.0})
    y_valid_pred = sess.run(y_pred, feed_dict={X: X_valid, keep_prob: 1.0})
    y_imb_pred = sess.run(y_pred, feed_dict={X: X_imb, keep_prob: 1.0})

y_count = Counter(y_train[:, 0])
print("Imbalance Rate: {}".format(y_count[0] / y_count[1]))
print("F1_score score on balanced training data: {:.4f}".format(f1_score(y_train, y_train_pred)))

y_count = Counter(y_valid[:, 0])
print("Imbalance Rate: {}".format(y_count[0] / y_count[1]))
print("F1_score score on balanced validation data: {:.4f}".format(f1_score(y_valid, y_valid_pred)))

y_count = Counter(y_imb)
print("Imbalance Rate: {}".format(y_count[0] / y_count[1]))
print("F1_score score on imbalance testing data: {:.4f}".format(f1_score(y_imb, y_imb_pred)))
