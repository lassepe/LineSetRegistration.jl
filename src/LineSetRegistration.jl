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

function run_test(
    observed_lines = TestDataGenerator.generate_lines(),
    spl_field = SPLFieldModel.SPLField(),
)
    fitted_line_tform, converged =
        IterativeClosestLine.fit_transformation(observed_lines, spl_field.lines)
    transformed_lines = map(fitted_line_tform, observed_lines)

    static_line_data = vcat(
        Visualizer.line_dataframe(spl_field.lines, "map"),
        Visualizer.line_dataframe(observed_lines, "initial"),
        Visualizer.line_dataframe(transformed_lines, "final_transformation"),
    )
    Visualizer.visualize(static_line_data) |> display
end

end # module
