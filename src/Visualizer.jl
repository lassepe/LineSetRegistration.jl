module Visualizer

using DataFrames: DataFrame, nrow
using VegaLite

function visualize(lines)
    line_data = line_dataframe(lines)
    color_property = hasproperty(line_data, :class) ? :class : nothing

    line_data |> @vlplot(
        :rule,
        width = 500,
        height = 300,
        x = {:x, scale = {domain = [-5, 5]}},
        y = {:y, scale = {domain = [-3, 3]}},
        x2 = :xend,
        y2 = :yend,
        color = color_property
    )
end

function line_dataframe(lines::DataFrame)
    lines
end

function line_dataframe(lines, class = nothing, i_iter = nothing)
    map(lines) do l
        (x, y), (xend, yend) = l
        line_data = (; x, y, xend, yend)

        line_data = if isnothing(class)
            line_data
        else
            merge(line_data, (; class))
        end

        line_data = if isnothing(i_iter)
            line_data
        else
            merge(line_data, (; i_iter))
        end
    end |> DataFrame
end

end # module
