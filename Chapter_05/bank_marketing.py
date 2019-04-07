import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score

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
bank_df.head()

X = bank_df.drop('y', axis=1).values
y = bank_df['y'].values.reshape(-1, 1)
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)
valid_X, test_X, valid_y, test_y = train_test_split(test_X, test_y, test_size=0.5, random_state=0)

n_features = train_X.shape[1]
m_train = train_X.shape[0]
n_output = 1
n_hidden = 32
learning_rate = 0.001
n_epochs = 20
batch_size = 128
drop_out = 0.4


with tf.name_scope('placeholders'):
    X = tf.placeholder(shape=(None, n_features), dtype=tf.float32)
    y = tf.placeholder(shape=(None, n_output), dtype=tf.float32)
    keep_prob = tf.placeholder(tf.float32)

with tf.name_scope('hidden_layer'):
    W = tf.Variable(tf.random_normal((n_features, n_hidden)))
    b = tf.Variable(tf.zeros((n_hidden,)))
    X_hidden = tf.nn.relu(tf.matmul(X, W) + b)
    X_hidden = tf.nn.dropout(X_hidden, keep_prob=keep_prob)

with tf.name_scope("output"):
    W = tf.Variable(tf.random_normal((n_hidden, n_output)))
    b = tf.Variable(tf.zeros((n_output,)))
    y_logit = tf.matmul(X_hidden, W) + b
    y_one_prob = tf.sigmoid(y_logit)
    y_pred = tf.round(y_one_prob)

with tf.name_scope("loss"):
    entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=y)
    l = tf.reduce_sum(entropy)

with tf.name_scope("optim"):
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(l)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    for epoch in range(1, n_epochs + 1):
        batch_pos = 0
        pos = 0
        while pos < m_train:
            X_batch = train_X[pos: pos + batch_size]
            y_batch = train_y[pos: pos + batch_size]
            feed_dict = {X: X_batch, y: y_batch, keep_prob: 0.5}
            _, loss = sess.run([train_op, l], feed_dict=feed_dict)
            print(f"epoch {epoch}, step {step}, loss: {loss}")
            step += 1
            pos += batch_size
    y_train_pred = sess.run(y_pred, feed_dict={X: train_X, keep_prob: 1.0})
    y_valid_pred = sess.run(y_pred, feed_dict={X: valid_X, keep_prob: 1.0})
    y_test_pred = sess.run(y_pred, feed_dict={X: test_X, keep_prob: 1.0})


print("Train accuracy score: {:.4f}".format(accuracy_score(train_y, y_train_pred)))
print("Validation accuracy score: {:.4f}".format(accuracy_score(valid_y, y_valid_pred)))
print("Test accuracy score: {:.4f}".format(accuracy_score(test_y, y_test_pred)))


print("Precision score on imbalance train data: {:.4f}".format(precision_score(train_y, y_train_pred)))
print("Recall score on imbalance train data: {:.4f}".format(recall_score(train_y, y_train_pred)))
print("ROC AUC score on imbalance train data: {:.4f}".format(roc_auc_score(train_y, y_train_pred)))
print("F1_score score on imbalance train data: {:.4f}".format(f1_score(train_y, y_train_pred)))


print("Precision score on imbalance valid data: {:.4f}".format(precision_score(valid_y, y_valid_pred)))
print("Recall score on imbalance valid data: {:.4f}".format(recall_score(valid_y, y_valid_pred)))
print("ROC AUC score on imbalance valid data: {:.4f}".format(roc_auc_score(valid_y, y_valid_pred)))
print("F1_score score on imbalance valid data: {:.4f}".format(f1_score(valid_y, y_valid_pred)))

print("Precision score on imbalance test data: {:.4f}".format(precision_score(test_y, y_test_pred)))
print("Recall score on imbalance test data: {:.4f}".format(recall_score(test_y, y_test_pred)))
print("ROC AUC score on imbalance test data: {:.4f}".format(roc_auc_score(test_y, y_test_pred)))
print("F1_score score on imbalance test data: {:.4f}".format(f1_score(test_y, y_test_pred)))
