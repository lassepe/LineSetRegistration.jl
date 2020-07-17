function fit_line_transformation(
    lines,
    map_lines;
    n_iterations_max = 50,
    snapshot_stepsize = show_debug_animation ? 1 : nothing,
    min_grad_norm = 1e-3,
    min_cost = 0.1,
    optimizer = LevenBergMarquardt(),
)
    θ::PoseTransformation{Float64} = zero(PoseTransformation)
    "Normalize transformation to rotate around com of lines."
    lines_com, lines_mass = center_of_mass(lines)
    debug_snapshots = []

    optimizer_state = initial_state(optimizer)
    grad_result = DiffResults.GradientResult(θ)

    function cost(params)
        tform = pose_transformation(params; rot_center = lines_com)
        c = sum(l -> line_fit_error(tform(l), map_lines), lines)
    end

    converged = false
    cost_decreased = true

    for i in 1:n_iterations_max
        println("outer_i: $i")
        grad_result = ForwardDiff.gradient!(grad_result, cost, θ)
        ∇θ = DiffResults.gradient(grad_result)
        V = DiffResults.value(grad_result)
        println("$grad_result")
        if V < min_cost
            converged = true
            break
        end
        if !cost_decreased
            break
        end

        θ, optimizer_state, cost_decreased =
            optimizer_step(optimizer, cost, V, θ, ∇θ, optimizer_state)

        # take a snapshot every few iterations
        if !isnothing(snapshot_stepsize) && iszero(i % snapshot_stepsize)
            tform_snapshot = pose_transformation(θ; rot_center = lines_com)
            cost_snapshot = cost(θ)
            push!(debug_snapshots, (; i, tform_snapshot, cost_snapshot))
        end
    end

    converged ? @info("Converged!") : @warn("Not converged!")

    pose_transformation(θ; rot_center = lines_com), converged, debug_snapshots
end

