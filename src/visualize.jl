#===================================== visualization backends =====================================#
#
abstract type DisplayBackend end
struct GGPlotBackend <: DisplayBackend end
struct VegaBackend <: DisplayBackend end

function visualize(lines; backend = GGPlotBackend())
    line_data = line_dataframe(lines)
    color_property = hasproperty(line_data, :class) ? :class : nothing
    visualize_(backend, line_data, color_property)
end

function visualize_(::GGPlotBackend, line_data, color_property)
    (
        ggplot(
            line_data,
            aes(x = :x, y = :y, xend = :xend, yend = :yend, color = color_property, group=nothing),
        ) +
        geom_segment() +
        coord_fixed(ratio = 1)
    )
end

function visualize_(::VegaBackend, line_data, color_property)
    line_data |> @vlplot(
        :rule,
        width = 900,
        height = 400,
        x = :x,
        y = :y,
        x2 = :xend,
        y2 = :yend,
        color = color_property
    )
end

#====================================== input data conversion =====================================#

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

#================================= compose the final optimization =================================#

function debug_viz(static_line_data, debug_snapshots)
    dynamic_line_data = mapreduce(vcat, debug_snapshots) do (i, tform_snapshot)
        transformed_lines = map(l -> transform(tform_snapshot, l), initial_lines)
        line_dataframe(transformed_lines, "optimization step", i)
    end

    animate(
        visualize(static_line_data) +
        geom_segment(data = dynamic_line_data) +
        transition_manual(:i_iter),
        nframes = nrow(dynamic_line_data),
        width = 1000,
        height = 500,
        units = "px",
    )
end
