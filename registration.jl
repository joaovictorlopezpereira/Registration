using LinearAlgebra
using Plots


# TODO: Implement best_indicator
# TODO: Implement translations


# Returns an orthogonal matrix Q such that Q is in argmin_Q ||QX - Y||
function procrustes(X, Y)
  U, D, Vt = svd(Y * X')
  Q = U * Vt
  return Q
end


# Returns an indicator matrix P such that P is in argmin_P ||X - YP||
function best_indicator(X, Y)
  nx = size(X, 2)
  ny = size(Y, 2)
  P = zeros(ny, nx)

  return P
end


# Returns an orthogonal matrix Q and an indicator matrix Q such that P,Q are (approximately) in argmin_{P,Q} ||QA - BP||
function point_matching(A, B;
                           iters = 100, # Defines "iters" as an optional argument which is set at 100 as default
                           orthogonal = I(size(1, A))) # Defines "orthogonal" as a optional argument which is set as the Identity matrix as default
  P = zeros(size(B, 2), size(A, 2))
  Q = orthogonal
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

  # Plots the first and second rows of each matrix as dots in R2
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