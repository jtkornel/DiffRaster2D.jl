using Flux

W_truth = [1 2 3 4 5;
            5 4 3 2 1]
b_truth = [-1.0; -2.0]
ground_truth(x) = W_truth*x .+ b_truth

x_train = [ 5 .* rand(5) for _ in 1:10_000 ]
y_train = [ ground_truth(x) + 0.2 .* randn(2) for x in x_train ]

model(x) = W*x .+ b

W0 = rand(2, 5)
b0 = rand(2)

W = copy(W0)
b = copy(b0)

function loss(x, y)
    ŷ = model(x)
    sum((y .- ŷ).^2)
end

opt = Descent(0.01)

train_data = zip(x_train, y_train)
ps = Flux.params(W, b)

for (x,y) in train_data
    gs = Flux.gradient(ps) do
      loss(x,y)
    end
    Flux.Optimise.update!(opt, ps, gs)
end
