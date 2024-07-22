
INeRF(source) = (x, y, z, up(3), right(3), forward(3)) -> (r, g, b, density)

Stokes(source) = (
    I0 = INeRF(source)(..., up(3) + 0º around forward , right(3) + 0º around forward , forward(3)),
    I45 = INeRF(source)(..., up(3) + 45º around forward , right(3) + 45º around forward , forward(3)),
    I90 = INeRF(source)(..., up(3) + 90º around forward , right(3) + 90º around forward , forward(3)),
    I135 = INeRF(source)(..., up(3) + 135º around forward , right(3) + 135º around forward , forward(3)),
) -> (
    S0 = 1/2 * (I0 + I45 + I90 + I135),
    S1 = I0 - I90,
    S2 = I45 - I135,
    S3 = 0
)

Mueller = (
    S(null) = Stokes(null),
    Sout = Stokes(0º),
    Sout = Stokes(45º),
    Sout = Stokes(90º),
    Sout = Stokes(135º),
) -> (
    M = resolver sistema Sout(0º|45º|90º|135º) - S(null) = M * Sin(0º|45º|90º|135º)
)


