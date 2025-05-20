using LinearAlgebra
using Plots
import Random


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


# Plots the first and second entries of each column of each matrix as dots in R2
function scatter_cols!(plt, matrices; alpha = 1)
  for (_, A) in enumerate(matrices)
    scatter!(plt, A[1, :], A[2, :], label = false, markeralpha = alpha, markersize = 10)
  end
end

# Returns a plot with matrices plotted
function matrices_plot(matrices)
  # Initializes plt as an empty plot
  plt = plot()

  scatter_cols!(plt, matrices)

  # Returns the plot
  return plt
end


# Returns a GIF with a visualization of each iteration of the point_matching function
function gif_point_matching(A, B; iters = 50, framerate = 1, gif_name = "point_matching.gif", images_name = "gif_")
  Q = I(size(A, 1))
  P = zeros(size(B, 2), size(A, 2))
  u = zeros(size(A, 1))

  # Compute the plots from the algorithm iterations
  plts = []
  for i in 1:iters
    P = best_indicator(Q*A .+ u, B)
    Q, u = best_rigid_transf(A, B*P)

    plt = plot()
    scatter_cols!(plt, [Q*A .+ u, B * P])
    scatter_cols!(plt, [B], alpha = 0.5)

    push!(plts, plt)

    print("iteration: $i\n")
    print("error: $(norm(Q*A .+ u - B*P))\n\n")
  end

  x_limits = (minimum(plt -> xlims(plt)[1], plts), maximum(plt -> xlims(plt)[2], plts))
  y_limits = (minimum(plt -> ylims(plt)[1], plts), maximum(plt -> ylims(plt)[2], plts))

  # Force x and y axis sizes to be equal, to maintain proportion
  if y_limits[2] - y_limits[1] < x_limits[2] - x_limits[1]
    y_limits = (y_limits[1], y_limits[1] + x_limits[2] - x_limits[1])
  else
    x_limits = (x_limits[1], x_limits[1] + y_limits[2] - y_limits[1])
  end

  zoom_out = 1.5
  x_avg = (x_limits[1] + x_limits[2]) / 2
  y_avg = (y_limits[1] + y_limits[2]) / 2
  x_limits = (x_avg - zoom_out * (x_limits[2] - x_limits[1]) / 2, x_avg + zoom_out * (x_limits[2] - x_limits[1]) / 2)
  y_limits = (y_avg - zoom_out * (y_limits[2] - y_limits[1]) / 2, y_avg + zoom_out * (y_limits[2] - y_limits[1]) / 2)


  anim = @animate for i in 1:iters
    plot(plts[i], xlims = x_limits, ylims = y_limits, aspect_ratio = :equal)
    savefig("$images_name" * "_" * "$i")
  end

  gif(anim, gif_name, fps = framerate)
end


# %->8-------------------------------------------------------------------------------------------8<-%
#
#                                        Area for testing!
#
# %->8-------------------------------------------------------------------------------------------8<-%

Random.seed!(1234)

# Returns a 2 by 2 rotation matrix given an angle in radians
rotate_2d = (θ) ->[cos(θ) -sin(θ); sin(θ)  cos(θ)]

# Defines smile faces
original_smile_face = [1.3 1.7 1.0 1.50 2.0 1.2 1.8; 1.0 1.0 0.5 0.25 0.5 0.3 0.3]
rotated_smile_face = rotate_2d(pi/4) * original_smile_face
translated_smile_face = original_smile_face .+ [6.1; 2.5]
noisy_translated_rotated_smile_face = rotate_2d(pi/3) * (hcat(original_smile_face, [original_smile_face + 0.05 * randn(2, size(original_smile_face,2)) for _ in 1:3]...)) .+ [-2.3; 7.2]

# Plots all smile faces
savefig(matrices_plot([original_smile_face, rotated_smile_face, translated_smile_face, noisy_translated_rotated_smile_face]), "all_faces.png")

# Tests the point matching heuristic
gif_point_matching(original_smile_face, rotated_smile_face; iters = 6, gif_name = "original_and_rotated.gif", images_name="rotated")
gif_point_matching(original_smile_face, translated_smile_face; iters = 6, gif_name = "original_and_translated.gif", images_name="translated")
gif_point_matching(original_smile_face, noisy_translated_rotated_smile_face; iters = 6, gif_name = "original_and_all.gif", images_name="noisy")
