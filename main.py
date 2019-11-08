from core import AntNetModel, AntDense


x1 = [i for i in range(10)]
x2 = [i for i in reversed(range(10))]

x = [x1, x2]
y = [1, 2]


ant = AntNetModel()
ant.add(AntDense(units=10, input_dim=10, activation="sigmoid"))
ant.add(AntDense(units=5, activation="sigmoid"))
ant.add(AntDense(units=1))

ant.compile(loss="mse")
ant.summary()

ant.predict(x1)
ant.predict(x2)
ant.predict(x)

ant.train(x, y, epochs=10)

ant.get_weights()