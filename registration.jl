using LinearAlgebra
using Plots


# Returns an orthogonal matrix Q such that ||QX - Y|| is minimized
function procrustes(X, Y)
  U, _, V = svd(Y * X')
  Q = U * V'
  return Q
end


# Returns an orthogonal matrix Q and some vector u such that ||(QX .+ u) - Y|| is minimized
# X and Y must have the same dimensions
function best_rigid_transf(X, Y)
  _, n = size(X)

  # Average of the columns
  x = 1/n * X * ones(n)
  y = 1/n * Y * ones(n)

  Q = procrustes(X .- x, Y .- y)
  u = y - Q * x

  return Q, u
end


# Returns an indicator matrix P such that ||X - YP|| is minimized
# (P is said to be an indicator matrix if it consists only of ones
# and zeros, with exactly one 1 per column)
function best_indicator(X, Y)
  nx, ny = size(X, 2), size(Y, 2)
  P = zeros(ny, nx)
  for i in 1:nx
    best_j = argmin([norm(X[:, i] - Y[:, j]) for j in 1:ny])
    P[best_j, i] = 1
  end
  return P
end


# Returns an orthogonal matrix Q, an indicator matrix P and a vector u such that ||QA .+ u - BP|| is approximately minimized
function point_matching(A, B;
                           iters = 100, # Defines "iters" as an optional argument which is set at 100 as default
                           orthogonal = I(size(A, 1))) # Defines "orthogonal" as a optional argument which is set as the Identity matrix as default
  P = zeros(size(B, 2), size(A, 2))
  Q = copy(orthogonal)
  u = zeros(size(A, 1))
  for _ in 1:iters
    Q, u = best_rigid_transf(A, B*P)
    P = best_indicator(Q*A .+ u, B)
  end
  return P, Q, u
end


# Returns a plot with matrices plotted
function matrices_plot(matrices)
  # Initializes plt as an empty plot
  plt = plot()

  # Plots the first and second entries of each column of each matrix as dots in R2
  for (_, A) in enumerate(matrices)
    scatter!(plt, A[1, :], A[2, :], label = false)
  end

  # Returns the plot
  return plt
end


# Returns a GIF with a visualization of each iteration of the point_matching function
function gif_point_matching(A, B; iters = 50, framerate = 1, gif_name = "point_matching.gif")
  Q = I(size(A, 1))
  P = zeros(size(B, 2), size(A, 2))
  u = zeros(size(A, 1))

  anim = @animate for i in 1:iters
    # Some variable is probably 'being lost' by calling point_matching each time, so the error will not change. If I had to guess, I'd say it is 'u'
    # P, Q, u = point_matching(A, B; iters=1, orthogonal=Q)

    Q, u = best_rigid_transf(A, B*P)
    P = best_indicator(Q*A .+ u, B)

    # TODO: Make xlims and ylims dynamic. If we don't set this value, then the plot will start changing its axis
    plot(matrices_plot([Q*A .+ u, B]), xlims=(-3,10), ylims=(-1,10)) # Maybe B*P should be plotted as well. However, there might be a lot of dots in the screen
    print("iteration: $i\n")
    print("error: $(norm(Q*A .+ u - B*P))\n\n")
  end

  gif(anim, gif_name, fps = framerate)
end


# %->8-------------------------------------------------------------------------------------------8<-%
#
#                                        Area for testing!
#
# %->8-------------------------------------------------------------------------------------------8<-%

# Returns a 2 by 2 rotation matrix given an angle in radians
rotate_2d = (θ) ->[cos(θ) -sin(θ); sin(θ)  cos(θ)]

# Defines smile faces
original_smile_face = [1.3 1.7 1.0 1.50 2.0 1.2 1.8; 0.8 0.8 0.5 0.25 0.5 0.3 0.3]
rotated_smile_face = rotate_2d(pi/4) * original_smile_face
translated_smile_face = original_smile_face .+ [6.1; 2.5]
noisy_translated_rotated_smile_face = rotate_2d(pi/3) * (hcat(original_smile_face, [original_smile_face + 0.05 * randn(2, size(original_smile_face,2)) for _ in 1:3]...)) .+ [-2.3; 7.2]

# Plots all smile faces
savefig(matrices_plot([original_smile_face, rotated_smile_face, translated_smile_face, noisy_translated_rotated_smile_face]), "all_faces.png")

# Tests the point matching heuristic
gif_point_matching(original_smile_face, rotated_smile_face; iters = 6, gif_name = "original_and_rotated.gif")
gif_point_matching(original_smile_face, translated_smile_face; iters = 6, gif_name = "original_and_translated.gif")
gif_point_matching(original_smile_face, noisy_translated_rotated_smile_face; iters = 6, gif_name = "original_and_all.gif")
