# Imports
# run `import Pkg; Pkg.add("$(PACKAGE_NAME)")` if package not installed already
using Flux, Statistics, BSON, Random, CUDA, ImageShow
using Flux: onecold, flatten
using Base.Iterators: repeated, partition
using Flux.Optimise: Optimiser, WeightDecay
using Logging: with_logger
using TensorBoardLogger: TBLogger, tb_overwrite, set_step!, set_step_increment!
using ProgressMeter: @showprogress
using Flux.Data: DataLoader
using MLDatasets, ImageInTerminal
using Flux: onehotbatch
using Flux.Losses: logitcrossentropy

# Struct to store hyperparameters
Base.@kwdef mutable struct Args
    eta = 1e-3              ## learning rate
    lambda = 1e-2           ## L2 regularizer param, implemented as weight decay
    batchsize = 128         ## batch size
    epochs = 10             ## number of epochs
    seed = 1                ## set seed > 0 for reproducibility
    use_cuda = true         ## if true use cuda (if available)
    infotime = 1 	        ## epoch interval for reporting
    checktime = 5           ## epoch interval to save model. 0 to disable
    tblogger = true         ## log training with tensorboard
    savepath = "models/"    ## filepath for saving models every checktime
end

# Function to get, reshape, encode, and batch data

function get_data(args)
    train_x, train_y = MLDatasets.MNIST(:train)[:]
    test_x, test_y = MLDatasets.MNIST(:test)[:]
    train_x, test_x = reshape(train_x, 28,28,1,:), reshape(test_x, 28,28,1,:)
    train_y, test_y = onehotbatch(train_y, 0:9), onehotbatch(test_y, 0:9)
    train_set = DataLoader((images=train_x, labels=train_y), batchsize=args.batchsize, shuffle=true)
    test_set = DataLoader((images=test_x, labels=test_y),  batchsize=args.batchsize)
    return train_set, test_set
end

# LeNet-5 Constructor 
function LeNet5(; imgsize=(28,28,1), num_classes=10) 
    out_conv_size = (div(imgsize[1],4) - 3, div(imgsize[2],4) - 3, 16)
    return Chain(
        Conv((5, 5), imgsize[end]=>6, relu),
        
        MaxPool((2, 2)),
        Conv((5, 5), 6=>16, relu),
        MaxPool((2, 2)),
        flatten,
        Dense(prod(out_conv_size), 120, relu), 
        Dense(120, 84, relu), 
        Dense(84, num_classes)
    )
end

# Utility functions
loss(y_hat, label) = logitcrossentropy(y_hat, label)
num_params(model) = sum(length, Flux.params(model)) 
round4(x) = round(x, digits=4)
function eval_loss_accuracy(loader, model, device)
    l = 0f0
    acc = 0
    ntot = 0
    for (x, y) in loader
        x, y = device(x), device(y)
        y_hat = model(x)
        l += loss(y_hat, y) * size(x)[end]        
        acc += sum(onecold(y_hat) .== onecold(y))
        ntot += size(x)[end]
    end
    return (loss = round4(l/ntot), acc = round4(acc/ntot*100))
end

# Training function
function train(; kws...)
    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)
    use_cuda = args.use_cuda && CUDA.functional()
    if use_cuda
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end
    ## DATA
    train_set, test_set = get_data(args)
    @info "Dataset Fashion-MNIST: $(size(train_set.data.images)[end]) train and $(size(test_set.data.images)[end]) test examples"
    ## MODEL AND OPTIMIZER
    model = device(LeNet5())
    @info "LeNet5 model: $(num_params(model)) trainable params"    
    ps = Flux.params(model)  
    opt = ADAM(args.eta) 
    if args.lambda > 0 ## add weight decay, equivalent to L2 regularization
        opt = Optimiser(WeightDecay(args.lambda), opt)
    end
    ## LOGGING
    if args.tblogger 
        tblogger = TBLogger(args.savepath, tb_overwrite)
        set_step_increment!(tblogger, 0) ## 0 auto increment since we manually set_step!
        @info "TensorBoard logging to \"$(args.savepath)\""
    end
    function report(epoch)
        train = eval_loss_accuracy(train_set, model, device)
        test = eval_loss_accuracy(test_set, model, device)        
        println("Epoch: $epoch   Train: $(train)   Test: $(test)")
        if args.tblogger
            set_step!(tblogger, epoch)
            with_logger(tblogger) do
                @info "train" loss=train.loss  acc=train.acc
                @info "test"  loss=test.loss   acc=test.acc
            end
        end
    end
    ## TRAINING
    @info "Starting training..."
    report(0)
    for epoch in 1:args.epochs
        @showprogress for (x, y) in train_set
            x, y = device(x), device(y)
            gs = Flux.gradient(ps) do
                    y_hat = model(x)
                    loss(y_hat, y)
                end
            Flux.Optimise.update!(opt, ps, gs)
        end
        ## Printing and logging
        epoch % args.infotime == 0 && report(epoch)
        if args.checktime > 0 && epoch % args.checktime == 0
            !ispath(args.savepath) && mkpath(args.savepath)
            modelpath = joinpath(args.savepath, "model.bson") 
            let model = cpu(model)
                BSON.@save modelpath model epoch
            end
            @info "Latest model saved to \"$(modelpath)\""
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end