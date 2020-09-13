module Optimizers

using LinearAlgebra: I

abstract type AbstractOptimizer end
function initial_state end
function optimizer_step end

Base.@kwdef struct LevenBergMarquardt <: AbstractOptimizer
    "The up scaling factor for adaptive damping (towards gradient step)"
    ρ_up::Float64 = 10
    "The down scaling factor for adaptive damping (towards gauss-newton step)"
    ρ_down::Float64 = 0.5
    "The initial regularization factor."
    λ₀::Float64 = 1.0
end

function initial_state(optimizer::LevenBergMarquardt)
    optimizer.λ₀
end

function step(optimizer::LevenBergMarquardt, Vfunc, V, θ, ∇θ, λ)
    # TODO: introduce line-search for dynamic damping.
    # levenberg-marquard step
    M = ∇θ * ∇θ'
    cost_decreased = false
    θ_candidate = θ
    for i in 1:10
        θ_candidate = θ - (M + λ * I) \ ∇θ * V
        V_candidate = Vfunc(θ_candidate)
        # if the cost cost_decreased, accept the step candidate
        cost_decreased = V_candidate < 0.99 * V
        if cost_decreased
            # accept step candidate
            break
        end
        # the step was not accepted, adjust the damping
        λ *= optimizer.ρ_up
    end
    return (; step = θ_candidate, λ = λ * optimizer.ρ_down, cost_decreased)
end

Base.@kwdef struct Descent <: AbstractOptimizer
    "The initial step size in negative gradient direction."
    step_size::Float64 = 0.01
    "The exponential decay factor with which the step size is multiplied at each iteration."
    step_decay::Float64 = 0.99
end

function initial_state(optimizer::Descent)
    optimizer.step_size
end

end # module
