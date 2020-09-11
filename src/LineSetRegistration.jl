module LineSetRegistration

include("Visualizer.jl")
include("Optimizers.jl")
include("GeometryTransformationUtils.jl")
include("SPLFieldModel.jl")
include("IterativeClosestLine.jl")
include("TestDataGenerator.jl")

import .IterativeClosestLine
import .SPLFieldModel
import .TestDataGenerator
import .Visualizer

function run_test(lines = TestDataGenerator.generate_lines(), spl_field = SPLFieldModel.SPLField())

    fitted_line_tform, converged, debug_snapshots =
        IterativeClosestLine.fit_transformation(lines, spl_field.lines)
    transformed_lines = map(fitted_line_tform, lines)

    static_line_data = vcat(
        Visualizer.line_dataframe(spl_field.lines, "map"),
        Visualizer.line_dataframe(lines, "initial"),
        Visualizer.line_dataframe(transformed_lines, "final_transformation"),
    )
    Visualizer.visualize(static_line_data) |> display

    lines, static_line_data, debug_snapshots
end

# initial_lines, static_line_data, debug_snapshots = run_test()

end # module