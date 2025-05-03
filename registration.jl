using LinearAlgebra
using Plots


# TODO: Implement best_indicator
# TODO: Implement translations


# Returns an orthogonal matrix Q such that ||QX - Y|| is minimized
function procrustes(X, Y)
  U, D, V = svd(Y * X')
  Q = U * V'
  return Q
end


# Returns an orthogonal matrix Q and some vector u such that ||(QX .+ u) - Y|| is minimized
# X and Y must have the same dimensions
# (TODO: Find proof that this algorithm is correct)
function best_rigid_transf(X, Y)
  m, n = size(X)

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
  nx = size(X, 2)
  ny = size(Y, 2)
  P = zeros(ny, nx)

  # For each column of X
  for i in 1:nx
    # Find the nearest column of Y
    best_j = -1
    best_dist = Inf
    for j in 1:ny
      d = norm(X[:, i] - Y[:, j])
      if d > best_dist
        continue
      end
      best_j = j
      best_dist = d
    end

    # Select it with P to minimize overall distance
    P[best_j, i] = 1
  end

  return P
end


# Returns an orthogonal matrix Q and an indicator matrix Q such that ||QA - BP|| is approximately minimized
function point_matching(A, B;
                           iters = 100, # Defines "iters" as an optional argument which is set at 100 as default
                           orthogonal = I(size(A, 1))) # Defines "orthogonal" as a optional argument which is set as the Identity matrix as default
  P = zeros(size(B, 2), size(A, 2))
  Q = copy(orthogonal)
  for i in 1:iters
    P = best_indicator(Q*A, B)
    Q = procrustes(A, B*P)
  end
  return P, Q
end


# Returns a plot with matrices plotted
function plot_matrices(matrices; return_plot=false)
  # Initializes plt as an empty plot
  plt = plot()

  # Plots the first and second entries of each column of each matrix as dots in R2
  for (i, A) in enumerate(matrices)
    scatter!(plt, A[1, :], A[2, :], label = false)
  end

  # Returns or displays the plot
  if (return_plot)
    return plt
  else
    display(plt)
  end
end


# Returns a GIF with a visualization of each iteration of the point_matching function
function gif_point_matching(A, B; iters = 50, framerate = 5)
  Q = I(size(A, 1))
  P = zeros(size(B, 2), size(A, 2))

  anim = @animate for i in 1:iters
    P, Q = point_matching(A, B; iters=1, orthogonal=Q)
    plt = plot_matrices([Q*A, B, B*P], return_plot=true)
    print("iteration: $i\n")
    print("error: $(norm(Q*A - B*P))\n\n")
  end # maybe "end" should be indented right below "for"?

  gif(anim, "point_matching.gif", fps = framerate)
end


# %->8-------------------------------------------------------------------------------------------8<-%
#
#                                        Area for testing!
#
# %->8-------------------------------------------------------------------------------------------8<-%

# # Matrix that represents a square
# M1 = [0.0 1.0 1.0 0.0;
#      0.0 0.0 1.0 1.0]

# # Matrix that represents (approximately) another square
# M2 = [2.0000  2.7071  2.0000  1.2929;
#      1.0000  1.7071  2.4142  1.7071]

# M3 = [2.00 2.35 2.70 2.35 2.00 1.65 1.29 1.65;
#      1.00 1.35 1.70 2.06 2.41 2.06 1.70 1.35]

# savefig(plot_matrices([M1, M2]; return_plot=true), "testing-1.png") # Working!
# savefig(plot_matrices([M1, M2, M3]; return_plot=true), "testing-2.png") # Working! (C is overwriting B in the plot, as intended)