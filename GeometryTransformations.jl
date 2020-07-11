module GeometryTransformations

using GeometryBasics: Line

function apply(transformation, line)
    p1_tformed, p2_tformed = transformation.(Tuple(line))
    Line(p1_tformed, p2_tformed)
end

end # module
