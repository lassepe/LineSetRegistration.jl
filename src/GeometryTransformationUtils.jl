module GeometryTransformationUtils

import CoordinateTransformations: AffineMap, LinearMap
using CoordinateTransformations: Translation
using GeometryBasics: Point, Line
using LinearAlgebra: ⋅, norm
using StaticArrays: FieldVector, SMatrix

# TODO: Is this considered type piracy?
function (tform::AffineMap)(l::Line)
    Line((tform.(l))...)
end

function (tform::LinearMap)(l::Line)
    Line((tform.(l))...)
end

struct PoseTransformation{T} <: FieldVector{3,T}
    x::T
    y::T
    α::T
end

function direction_vector(line::Line)
    p1, p2 = line
    p2 - p1, p1, p2
end

function length_sq(line::Line)
    line_vec, _ = direction_vector(line)
    line_vec ⋅ line_vec
end

function pose_transformation(transformation_parameters; kwargs...)
    pose_transformation(
        transformation_parameters[1],
        transformation_parameters[2],
        transformation_parameters[3];
        kwargs...,
    )
end

function pose_transformation(x, y, α; rot_center = zero(Point{2}))
    sα, cα = sin(α), cos(α)
    Translation(rot_center) ∘ Translation(x, y) ∘ LinearMap(SMatrix{2,2}(cα, sα, -sα, cα)) ∘
    inv(Translation(rot_center))
end

function center_of_mass(lines::AbstractVector{Line{N,T}}) where {N,T}
    total_com, total_mass = reduce(lines; init = (zero(Point{N,T}), 0.0)) do (cum_com, cum_mass), l
        p1, p2 = l
        line_vec = p2 - p1
        mass = norm(line_vec)
        com = p1 + 1 // 2 * line_vec
        new_cum_mass = cum_mass + mass
        @assert new_cum_mass > 0
        new_com = (cum_com * cum_mass + com * mass) / new_cum_mass
        (new_com, new_cum_mass)
    end
    total_com, total_mass
end

end # module
