
include("SPLFieldModel.jl")
import .SPLFieldModel

include("Visualizer.jl")
import .Visualizer

include("IterativeClosestLine.jl")
import .IterativeClosestLine

include("Optimizers.jl")
import .Optimizers

include("IterativeClosestLine.jl")
import .IterativeClosestLine

include("TestDataGenerator.jl")
import .TestDataGenerator

function run_test(lines = generate_test_lines(), spl_field = SPLFieldModel.SPLField())

    fitted_line_tform, converged, debug_snapshots =
        fit_line_transformation(lines, spl_field.lines)
    transformed_lines = map(fitted_line_tform, lines)

    static_line_data = vcat(
        line_dataframe(spl_field.lines, "map"),
        line_dataframe(lines, "initial"),
        line_dataframe(transformed_lines, "final_transformation"),
    )
    visualize(static_line_data) |> display

    lines, static_line_data, debug_snapshots
end

initial_lines, static_line_data, debug_snapshots = run_test()
