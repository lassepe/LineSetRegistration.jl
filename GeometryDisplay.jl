module GeometryDisplay

using GeometryBasics: GeometryBasics
using RCall, DataFrames
@rlibrary ggplot2

export visualize

function visualize(lines::AbstractVector{<:GeometryBasics.Line}; file = "out.png")
    # map lines to a visualizable dataframe
    line_data = map(lines) do l
        (x, y), (xend, yend) = GeometryBasics.decompose(GeometryBasics.Point, l)
        (; x, y, xend, yend)
    end |> DataFrame # NOTE: there is probably a way to avoid the construction of a DataFrame here.

    ggplot(line_data, aes(x=:x, y=:y, xend=:xend, yend=:yend)) + geom_segment()
end

end #module
