module GeometryDisplay

using Makie: Makie
using CairoMakie: CairoMakie
using GeometryBasics: GeometryBasics

function visualize(lines::AbstractVector{<:GeometryBasics.Line}; file = "out.png")
    line_points = map(lines) do l
        p1, p2 = GeometryBasics.decompose(GeometryBasics.Point, l)
        p1 => p2
    end
    CairoMakie.save(file, Makie.linesegments(line_points))
end

end #module
