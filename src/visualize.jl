#===================================== visualization backends =====================================#
#
abstract type DisplayBackend end
struct GGPlotBackend <: DisplayBackend end
struct VegaBackend <: DisplayBackend end

function visualize(lines; backend = GGPlotBackend())
    visualize_(backend, lines)
end

function visualize_(::GGPlotBackend, lines)
    line_data = line_dataframe(lines)
    color_property = hasproperty(line_data, :class) ? :class : nothing
    ggplot(line_data, aes(x = :x, y = :y, xend = :xend, yend = :yend, color = color_property)) +
    geom_segment()
end

function visualize_(::VegaBackend, lines)
    lines |>
    line_dataframe |>
    @vlplot(:rule, width = 900, height = 400, x = :x, y = :y, x2 = :xend, y2 = :yend)
end

#====================================== input data conversion =====================================#

function line_dataframe(lines::DataFrame)
    lines
end

function line_dataframe(lines, class=nothing)
    map(lines) do l
        (x, y), (xend, yend) = l
        line_data = (; x, y, xend, yend)
        if isnothing(class)
            line_data
        else
            merge(line_data, (; class))
        end
    end |> DataFrame
end
