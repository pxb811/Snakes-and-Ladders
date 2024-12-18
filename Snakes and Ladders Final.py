# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 20:02:44 2020

@author: Przemyslaw Bujak
"""

'''Note to marker: Some graphs in the report where produced with separate code. That code is not 
submitted as it was very rough and its only purpose was to produce the required graphs.'''

import numpy as np
import matplotlib.pyplot as plt
import sys

def make_roll(no_d6):
    '''
    Simulate rolling no_d6 fair six-sided dice.
    np.random.randit(low, high=None, size=None, dtype='l') returns random integers from the 
    “discrete uniform” distribution of the specified dtype in the “half-open” interval [low, high)

    Parameters
    ----------
    no_d6 : int
        Number of d6s to roll.

    Returns
    -------
    roll : int
        The result of the roll of no_d6 d6s.

    '''
    
    # Use a for in range(no_d6) loop to simulate no_d6 rolls.
    roll = 0
    for i in range(no_d6):
        roll += np.random.randint(1, 7)
    return roll

def calculate_Monte_Carlo(simulations, final_square, snakes_ladders, win_condition, no_d6=1):
    '''
    Perform the Monte Carlo method for simulations simulations on the board defined by final_square
    and snakes_ladders with a d6 and the 'over' win condition (need to reach final_square or 
    beyond).

    Parameters
    ----------
    simulations : int
        The number of times we simulate the game.
    final_square : int
        The final square on the board.
    snakes_ladders : array_like
        Array of tuples containing (no_ladders + no_snakes) tuples of the form [start, end] squares
        of the snakes and ladders.

    Returns
    -------
    None.

    '''
    
    # To order the squares according to the average length of time until victory, create an array
    # that holds all the visited_squares.
    # Create an array consisting of [square, duration], then will have to flatten it to
    # [square, mean_duration], where duration is the number of turns until victory.
    # Make the array include all the squares on the board even though some of them may
    # not be visited, so that we know the exact size of the array and so can use numpy.
    
    # square_duration is a numpy array with final_square rows and 3 columns, where each row is
    # [square, sum of durations, number of times we have added to the sum = number of times
    # that this square has been visited].
    # The third column is a counter so that we can find the mean duration by doing
    # sum of durations/ number of times we have added to the sum.
    
    # Only the while loop differs depending on the win_condition, but it is easier to repeat the 
    # entire for loop.
    if win_condition == 'Over':
        # Create empty square_duration array, where the first column lists all the squares from 
        # zero to final_square.
        square_duration = np.zeros((final_square, 3), dtype=int)
        square_duration[:, 0] = range(final_square)          
        
        turns_played_list_snakes_ladders = np.array([], dtype=int)
        for i in range(simulations):
            # Reset variables at the start of a new simulation of a game
            visited_squares = np.array([], dtype=int)
            current_square = 0
            turns_played = 0
            
            while current_square < final_square:
                # visited_squares includes 0 but not 100 (or above)
                visited_squares = np.append(visited_squares, current_square)
                ''' 
                    np.random.randit(low, high=None, size=None, dtype='l') returns random integers
                    from the “discrete uniform” distribution of the specified dtype in the 
                    “half-open” interval [low, high).
                              '''
                roll = make_roll(no_d6)
                current_square += roll
                # Compare current position with array of snakes and ladders.
                # If current position is at the start of a snake/ladder then set current position to 
                # end of snake/ladder.
                for chute in snakes_ladders:
                    if current_square == chute[0]:
                        current_square = chute[1]
    
                turns_played += 1
            
            turns_played_list_snakes_ladders = np.append(turns_played_list_snakes_ladders, turns_played)
            
            for turn, square in enumerate(visited_squares):
                square_duration[square][1] += turns_played-turn
                square_duration[square][2] += 1
        
    elif win_condition == 'Exact':
        square_duration = np.zeros((final_square, 3), dtype=int)
        square_duration[:, 0] = range(final_square)        
        
        turns_played_list_snakes_ladders = np.array([], dtype=int)
        for i in range(simulations):
            # Reset variables at the start of a new simulation of a game
            visited_squares = np.array([], dtype=int)
            current_square = 0
            turns_played = 0
            
            while current_square != final_square:
                # visited_squares includes 0 but not 100 (or above)
                visited_squares = np.append(visited_squares, current_square)
                roll = np.random.randint(1, 7)
                
                # Allow the roll if player stays on the board.
                if current_square + roll <= final_square:
                    current_square += roll
                    # Compare current position with array of snakes and ladders.
                    # If current position is at the start of a snake/ladder then set current position 
                    # to end of snake/ladder.
                    for chute in snakes_ladders:
                        if current_square == chute[0]:
                            current_square = chute[1]
                # Else: do not update the current_square - just add to the turns_played.
                turns_played += 1
            
            turns_played_list_snakes_ladders = np.append(turns_played_list_snakes_ladders, turns_played)
            
            for turn, square in enumerate(visited_squares):
                square_duration[square][1] += turns_played-turn
                square_duration[square][2] += 1
                
    elif win_condition == 'Bounce':
        square_duration = np.zeros((final_square, 3), dtype=int)
        square_duration[:, 0] = range(final_square)          
        
        turns_played_list_snakes_ladders = np.array([], dtype=int)
        for i in range(simulations):
            # Reset variables at the start of a new simulation of a game
            visited_squares = np.array([], dtype=int)
            current_square = 0
            turns_played = 0
            
            while current_square != final_square:
                # visited_squares includes 0 but not 100 (or above)
                visited_squares = np.append(visited_squares, current_square)

                roll = np.random.randint(1, 7)
                # Allow the roll if player stays on the board.
                if current_square + roll <= final_square:
                    current_square += roll
                # Bounce back if roll beyond the final_square.
                elif current_square + roll > final_square:
                    current_square = final_square - np.abs((roll - (final_square - current_square)))
                # Else: display error.
                else:
                    print('ERROR: current_square > final_square!')
                    sys.exit()
                    
                # Compare current position with array of snakes and ladders.
                # If current position is at the start of a snake/ladder then set current position 
                # to end of snake/ladder.
                for chute in snakes_ladders:
                    if current_square == chute[0]:
                        current_square = chute[1]
                    
                turns_played += 1
            
            turns_played_list_snakes_ladders = np.append(turns_played_list_snakes_ladders, turns_played)
            
            for turn, square in enumerate(visited_squares):
                square_duration[square][1] += turns_played-turn
                square_duration[square][2] += 1
    
    else:
        print('ERROR: Invalid win_condition!')
        sys.exit()

        
    print(f'square_duration = {square_duration}')
    
    # To avoid dividing by zero, set values for start snakes_ladders squares to end squares.
    for i, k in enumerate(snakes_ladders):
        start = k[0]
        end = k[1]
        # We need to deal with ladders that end on final_square.
        if end == final_square:
            square_duration[start][1:] = [0, 1]
            continue
        square_duration[start][1:] = square_duration[end][1:]
    
    print(f'square_duration = {square_duration}')
    
    # NOTE: We can still end up dividing by zero if a square is never visited in any of the 
    # simulations, but we make this very unlikely by limiting the minimum number of simulations
    # to 100.
    
    # square_mean_duration consists of [square, mean_duration].
    # Need to transpose using .T to get the correct shape.
    square_mean_duration = np.array([square_duration[:, 0], 
                                     square_duration[:, 1]/square_duration[:, 2]]).T
    print(f'square_mean_duration = {square_mean_duration}')
    
    # Order the squares according to the average length of time until victory.
    square_mean_duration_ordered = order_square_mean_duration(square_mean_duration)
    print(f'square_mean_duration_ordered = {square_mean_duration_ordered}')
    
    # Use np.mean() and np.var() to find the variance and mean of the duration from square zero.
    mean_turns_played = np.mean(turns_played_list_snakes_ladders)
    var_turns_played = np.var(turns_played_list_snakes_ladders)
    print('\nFor a snakes & ladders board:')
    print(f'\nmean_turns_played = {mean_turns_played}\nvar_turns_played = {var_turns_played}')
    print(f'Minimum game length = {np.amin(turns_played_list_snakes_ladders)}')
    
    # Plot mean_duration against square.
    plot_it(x=square_mean_duration[:, 0],
                   ys=np.array([square_mean_duration[:, 1]]), 
                   y_labels=np.array(['square_mean_duration']), x_name='Square',
               y_name='Average number of turns until victory',
               plot_title='Plot of average duration until victory against square',
               individual_points=False)
        
def Monte_Carlo(board_parameters, win_condition):
    simulations = get_number_of_simulations()    
    calculate_Monte_Carlo(simulations, *board_parameters, win_condition, no_d6=1)
    
def order_square_mean_duration(square_mean_duration):
    '''
    Order square_mean_duration from shortest to longest mean_duration, showing the appropriate
    square number.

    Parameters
    ----------
    square_mean_duration : array_like
        Array of shape (final_square, 2) of the form [square, mean_duration for that square] going 
        from square 0 (off the board) to square (final_square - 1).

    Returns
    -------
    square_mean_duration_ordered : ndarray
        Array of the same shape as square_mean_duration of the form 
        [square, mean_duration for that square] ordered from longest to shortest mean_duration.

    '''
    
    # Using np.sort() along axis=0 only orders the mean_durations and keeps the squares in their
    # original order (they are already sorted).
    # We therefore use np.argsort() to return the array of indices of ordered data
    # and then reconstruct the original array in the sorted order.
    ordered_indices = np.argsort(square_mean_duration, axis=0)
    
    # We need to use the second (index=1) column of ordered_indices.
    square_mean_duration_ordered = square_mean_duration[ordered_indices[:, 1]]
    return square_mean_duration_ordered

def plot_it(x, ys, y_labels, x_name='x', y_name='y', plot_title='Plot', individual_points=False):
    '''
    Plot many different ys versus the same x on the same axes, graph, and figure.
    
    ---------------------
    Parameters:
    ---------------------
    
    x : array_like
        The independent variable to be plotted on be x-axis.
    
    ys : array_like
        Array of dependent variables to be plotted on be y-axis, where each row of ys is an array
        of y-values to plo against x.
    
    x_name : string
        The name on the x-axis.
    
    y_name : string
        The name on the y-axis.
        
    y_labels : array_like
        Array of size np.shape(ys)[0] where each element y_labels[i] corresponds to ys[i].
    
    plot_title : string
        The title on the graph.
        
    individual_points : Boolean, optional
        If True, will plot individual points as 'r.'. The default is False.
    
    --------------------
    Returns:
    --------------------
    
    figure : matplotlib.figure.Figure
        The plot.
    '''
    
    # Plot
    figure = plt.figure(figsize=(10,6))

    plt.title(plot_title, fontsize=16)
    plt.xlabel(x_name, fontsize=16)
    plt.ylabel(y_name, fontsize=16)
    for i, k in enumerate(ys):
        plt.plot(x, k, label=y_labels[i])
        # Useful to plot individual points for mean duration against square.
        if individual_points:
            plt.plot(x, k, 'r.')
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.show()
    return figure

def generate_P0_over_win_condition_2d6(final_square):
    ######################################
    # Work in progress
    ######################################
    '''
    Generate a stochastic matrix for an empty board of size final_square x final_square and rolling
    a single fair d6 and the 'over' win condition (need to reach final_square or beyond).

    Parameters
    ----------
    final_square : int
        The final square on the board.

    Returns
    -------
    P0 : ndarray
        Array of size (final_square+1)x(final_square+1) (because the game starts off the board on 
        square zero) made up entirely of zeros apart from the last element on the main diagonal 
        and the six elements to the right of the main diagonal, which equal 1/6, apart from the 
        last column which is made so that each row sums to 1.

    '''
    
    # Create empty (N+1)x(N+1) array
    P0 = np.zeros((final_square+1, final_square+1))
    # We set the final element = 1 so that it stays in the final state
    P0[-1][-1] = 1
    # Set the six elements to the right of the main diagonal = 1/6
    # Enumerate loops through each row
    for i, k in enumerate(P0):
        k[i+1:i+7] = 1/6
        # The edge of the matrix (the last column) must go as 1/6, 2/6, 3/6, 4/6, 5/6, 1, 1 for the
        # final 7 rows (np.size(P0, axis=0) gives the number of rows in P0)
        if np.size(P0, axis=0)-1 > i >= np.size(P0, axis=0)-7:
            # This will happen for the last bar one 6 rows
            k[-1] *= 6 - (np.size(P0, axis=0)-2-i)
    return P0

def generate_P0_over_win_condition(final_square):
    '''
    Generate a stochastic matrix for an empty board of size final_square x final_square and rolling
    a single fair d6 and the 'over' win condition (need to reach final_square or beyond).

    Parameters
    ----------
    final_square : int
        The final square on the board.

    Returns
    -------
    P0 : ndarray
        Array of size (final_square+1)x(final_square+1) (because the game starts off the board on 
        square zero) made up entirely of zeros apart from the last element on the main diagonal 
        and the six elements to the right of the main diagonal, which equal 1/6, apart from the 
        last column which is made so that each row sums to 1.

    '''
    
    # Create empty (N+1)x(N+1) array
    P0 = np.zeros((final_square+1, final_square+1))
    # We set the final element = 1 so that it stays in the final state
    P0[-1][-1] = 1
    # Set the six elements to the right of the main diagonal = 1/6
    # Enumerate loops through each row
    for i, k in enumerate(P0):
        k[i+1:i+7] = 1/6
        # The edge of the matrix (the last column) must go as 1/6, 2/6, 3/6, 4/6, 5/6, 1, 1 for the
        # final 7 rows (np.size(P0, axis=0) gives the number of rows in P0)
        if np.size(P0, axis=0)-1 > i >= np.size(P0, axis=0)-7:
            # This will happen for the last bar one 6 rows
            k[-1] *= 6 - (np.size(P0, axis=0)-2-i)
    return P0

def generate_P0_exact_win_condition(final_square):
    '''
    Generate a stochastic matrix for an empty board of size final_square x final_square and rolling
    a single fair d6 and the 'exact' win condition (stay put if roll above final_square).

    Parameters
    ----------
    final_square : int
        The final square on the board.

    Returns
    -------
    P0 : ndarray
        Array of size (final_square+1)x(final_square+1) (because the game starts off the board on 
        square zero) made up entirely of zeros apart from the last element on the main diagonal 
        and the six elements to the right of the main diagonal, which equal 1/6, apart from the 
        last column which is made so that each row sums to 1.

    '''
    
    # Create empty (N+1)x(N+1) array
    P0 = np.zeros((final_square+1, final_square+1))
    # We set the final element = 1 so that it stays in the final state
    P0[-1][-1] = 1
    # Set the six elements to the right of the main diagonal = 1/6
    # Enumerate loops through each row
    for i, k in enumerate(P0):
        k[i+1:i+7] = 1/6
        # Rolls off the board are equivalent to staying in the same place, so add the remainder of 
        # the probabilities for that row from 1 to the diagonal.
        if np.size(P0, axis=0)-1 > i > np.size(P0, axis=0)-7:
            # This will happen for the last bar one 5 rows.
            # Set the diagonal equal to 1 - np.sum(k). 
            k[i] = 1 - np.sum(k)
    return P0

def generate_P0_bounce_win_condition(final_square):
    '''
    Generate a stochastic matrix for an empty board of size final_square x final_square and rolling
    a single fair d6 and the 'bounce' win condition (if overshoot the final_square, bounce back
    the number of squares that was overshot, i.e. if roll 3 from square 98, go to square 99).

    Parameters
    ----------
    final_square : int
        The final square on the board.

    Returns
    -------
    P0 : ndarray
        Array of size (final_square+1)x(final_square+1) (because the game starts off the board on 
        square zero) made up entirely of zeros apart from the last element on the main diagonal 
        and the six elements to the right of the main diagonal, which equal 1/6, apart from the 
        last column which is made so that each row sums to 1.

    '''
    
    # Create empty (N+1)x(N+1) array
    P0 = np.zeros((final_square+1, final_square+1))
    # We set the final element = 1 so that it stays in the final state
    P0[-1][-1] = 1
    # Set the six elements to the right of the main diagonal = 1/6
    # Enumerate loops through each row
    for i, k in enumerate(P0):
        k[i+1:i+7] = 1/6
        # Rolls off the board are equivalent to staying in the same place, so add the remainder of 
        # the probabilities for that row from 1 to the diagonal.
        if np.size(P0, axis=0)-1 > i > np.size(P0, axis=0)-7:
            # This will happen for the last bar one 5 rows.
            # The [:-1] excludes the final_square, because cannot bounce back onto the final_square.
            # Under this if statement, i starts at np.size(P0, axis=0)-7+1 = np.size(P0, axis=0)-6.
            # We want the first iteration to add 1/6 to k[-2:-1], so use 
            # k[-1 - (i + 7 - np.size(P0, axis=0)):-1] = k[-1 - ((-6) + 7):-1] = k[-2:-1].
            k[-1 - (i + 7 - np.size(P0, axis=0)):-1] += 1/6
    return P0

def calculate_probabilities(final_square, n, P):
    '''
    Calculate the probability of reaching the final_square on the board based on which matrix P was
    generated in range(n+1) turns or fewer, and in exactly range(n+1) turns.

    Parameters
    ----------
    final_square : int
        The final square of the board based on which matrix P was generated.
    n : int
        The range of the number of turns for which we calculate the probabilities.
    P : array_like
        The stochastic matrix for which we calculate the probabilities.

    Returns
    -------
    p_n_column_list : ndarray
        Array of size (final_square+1)x(n+1) where each element (p_n)_{ij} of the matrix 
        p_n_column_list gives the probability of reaching the final square from square i after j 
        turns OR FEWER (0<=i<=final_square, j>=1). 
        Note that p_n_column_list[:, 0] gives the column probabilities after ONE turn!
    pi_n_column_list : ndarray
        Array of the same size as p_n_column_list where each element (pi_n)_{ij} of the matrix 
        pi_n_column_list gives the probability of reaching the final square from square i after 
        EXACTLY j turns (0<=i<=final_square, j>=1).
    pi_n_column_list_times_r : ndarray
        Array of the same size as pi_n_column_list where each element (pi_n_r)_{ir} of the matrix 
        pi_n_column_list_times_r equals the element (pi_n)_{ij}*r (j, r >= 1), where (pi_n)_{ij} is
        an element of the matrix pi_n_column_list.

    '''
    
    # Note that we can use just the stochastic matrix to the n and then take the final column to 
    # obtain the probability of victory in n turns starting from any square.
    # Do this now to try and order the squares according to the average number of turns taken until 
    # victory.
    
    # (P_n)^1 = P
    P_n = P
    
    # Each element (p_n)_{ij} of the matrix p_n_column_list gives the probability of reaching the 
    # final square from square i after j turns OR FEWER (0<=i<=final_square, j>=1).
    # Note that p_n_column_list[:, 0] gives the column probabilities after ONE turn!
    p_n_column_list = np.zeros((final_square+1, n+1))
    p_n_column_list[:, 0] = P[:, -1]
    
    for i in range(n):
        P_n = np.matmul(P_n, P)
        p_n_column_list[:, i+1] = P_n[:, -1]
    
    # Now for probability of EXACTLY j turns
    pi_n_column_list = np.zeros(np.shape(p_n_column_list))
    pi_n_column_list_times_r = np.zeros(np.shape(pi_n_column_list))
    # Iterate over all the columns in p_n_column_list
    for i in range(np.size(p_n_column_list, axis=1)):
        if i == 0:
            pi_n_column_list[:, i] = p_n_column_list[:, i] - 0
        else:
            pi_n_column_list[:, i] = p_n_column_list[:, i] - p_n_column_list[:, i-1]
        # Multiply each element pi_r by r
        # (We multiply by (i+1) instead of i because the index=0 column is the probability after 
        # ONE turn)
        pi_n_column_list_times_r[:, i] = (i+1)*pi_n_column_list[:, i]
    return p_n_column_list, pi_n_column_list, pi_n_column_list_times_r

def calculate_mean_duration(board_parameters, pi_n_column_list_times_r):
    '''
    Calculate the average number of turns it takes to reach the final_square from any square, 
    excluding the final_square (which is obviously zero so we omit it). Omiting final_square makes
    mean_duration have the same size as t from the Fundamental Matrix so that we can compare them.
    Manually set the mean_duration for any squares that are the starting_square of a ladder that 
    ends on final_square to zero.

    Parameters
    ----------
    board_parameters : array_like
        Array containing (final_square, snakes_ladders).
    pi_n_column_list_times_r : array_like
        Array of the same size as pi_n_column_list where each element (pi_n_r)_{ir} of the matrix 
        pi_n_column_list_times_r equals the element (pi_n)_{ij}*r (j, r >= 1), where (pi_n)_{ij} is
        an element of the matrix pi_n_column_list.

    Returns
    -------
    average_turns_for_victory_column_cut : ndarray
        Array of size final_square where each element average_turns_for_victory_column_cut[i] gives
        the average number of turns it takes to reach the final_square from square i.

    '''
    
    # The average_turns_for_victory is the sum of the rows of the matrix pi_n_column_list_times_r
    average_turns_for_victory_column = np.sum(pi_n_column_list_times_r, axis=1)
    
    # Manually set final_square duration to zero.
    average_turns_for_victory_column[-1] = 0
    
    # Manually set the mean_duration for any squares that are the starting_square of a ladder that 
    # ends on final_square to zero. 
    final_square, snakes_ladders = board_parameters[0], board_parameters[1]
    # First if statement to make code more efficient by not running it when not necessary.
    if final_square in snakes_ladders:
        for i, k in enumerate(snakes_ladders):
            # We need to deal with ladders that end on final_square.
            if k[1] == final_square:
                average_turns_for_victory_column[k[0]] = 0

    # Each element average_turns_for_victory_column[i] gives the average number of turns it takes
    # to reach the final_square from square i.
    # The above works for all squares apart from final_square, which gives 1 (it should obviously 
    # be zero). This comes from the fact that p_n_column_list must include final_square to have 
    # the same shape as P. The probability of going from final_square to final_square in 1 turn
    # is equal to 1, so when we sum pi_n_column_list_times_r we get 1.
    # We could manually set it to zero, or simply cut off the final_square element.
    # We choose the latter option so that we can compare it to t from the fundamental matrix.
    average_turns_for_victory_column_cut = average_turns_for_victory_column[:-1]
    return average_turns_for_victory_column

def generate_P(final_square, snakes_ladders, win_condition):
    '''
    Generate a stochastic matrix P for a given board, array of snakes and ladders, win condition,
    and rolling a single fair d6.

    Parameters
    ----------
    final_square : int
        The final square on the board. Determines the size of P, which is given by 
        (final_square+1)x(final_square+1).
    snakes_ladders : array_like
        Array of tuples containing (no_ladders + no_snakes) tuples of the form [start, end] squares
        of the snakes and ladders.
    win_condition : str
        Option chosen in the win_condition_menu() that determines when a game of Snakes & Ladders
        ends.

    Returns
    -------
    P : ndarray
        The stochastic (transition) matrix.

    '''
    
    # We find the stochastic matrix for the snakes & ladders board, P, by performing the following 
    # operations on the stochastic matrix for the empty board, P0, 
    # (i) add the \muth column to the \nuth column,
    # (ii) make the \muth column zero,
    # (iii) set the \muth row equal to the \nuth row,
    # where (\mu, \nu) are the (start, end) of the snakes and ladders.
    
    # Generate the stochastic matrix for the empty board, P0.
    if win_condition == 'Over':
        P = generate_P0_over_win_condition(final_square)
    elif win_condition == 'Exact':
        P = generate_P0_exact_win_condition(final_square)
    elif win_condition == 'Bounce':
        P = generate_P0_bounce_win_condition(final_square)
    else:
        print('ERROR: Invalid win_condition!')
        sys.exit()
    
    for chute in snakes_ladders:
        P[:, chute[1]] += P[:, chute[0]]
        P[:, chute[0]] = 0
        
        # To make \mu and \nu behave as essentially the same square, set the \muth row equal to the
        # \nuth row.
        P[chute[0], :] = P[chute[1], :]
    return P

def generate_random_board_do_not_allow_same_start_end():
    '''
    Generate a board with a random number of 
    squares, snakes, and ladders, and random start and end squares for the snakes and ladders 
    under the conditions that no more than one snake/ladder can start on the same square AND that
    a single square cannot have a start and an end of a chute. This means that any individual 
    square can ONLY either have
    (i) nothing,
    (ii) ONE start of a chute,
    (iii) one OR more ENDS of chutes.

    Returns
    -------
    final_square : int
        The final square on the board.
    snakes_ladders : ndarray
        Array of tuples containing (no_ladders + no_snakes) tuples of the form [start, end] squares
        of the snakes and ladders.

    '''
    
    # We choose a reasonable, yet arbitrary, range for each randomisation process.
    # Random board size
    final_square = np.random.randint(50, 151)
    # We choose a random number of snakes and ladders from an arbitrarily chosen range.
    # Note that we have a different range for snakes and ladders so that the asymmetry can help us 
    # check if our code is behaving as intended, as well as to produce more interesting results.
    no_ladders = np.random.randint(5, 16)
    no_snakes = np.random.randint(4, 15)
    
    print(f'final_square = {final_square} \nno_ladders = {no_ladders} \nno_snakes = {no_snakes}')
    
    # We need a running list of starts and ends of chutes to compare to so that we do NOT generate
    # a chute that 
    # (i) starts on a square that another chute STARTS at,
    # (ii) starts on a square that another chute ENDS at,
    # (ii) ends on a square that another chute starts at.
    # Therefore, we allow chutes to end on the same square.
    
    # We use np.arange instead of np.zeros to make starting_squares unique, and add final_square+1
    # so that it is out of the range of possible starting_squares (and end_squares).
    starting_squares = np.arange(no_ladders + no_snakes, dtype=int) + final_square+1
    end_squares = starting_squares.copy()
    # i is the number of successful generations.
    i = 0
    
    while i < no_ladders + no_snakes:
        # First do ladders.
        if i < no_ladders:
            # The first ladder can always start in the range [1, final_square), because there is no
            # conflict with other chutes (as so far no other chutes exist).
            starting_squares[i] = np.random.randint(1, final_square)
            # For ladders starting_squares[i] < end_squares[i] <= final_square.
            end_squares[i] = np.random.randint(starting_squares[i]+1, final_square+1)
            # The first condition checks that starting_squares is unique, which together with the
            # second condition ensures conditions (i) and (ii) from above.
            # The third condition ensures condition (iii) from above.
            if not np.any(np.unique(starting_squares, return_counts=True)[-1]-1) \
                and starting_squares[i] not in end_squares \
                and end_squares[i] not in starting_squares:
                i += 1
        # Now do snakes.
        else:
            # Snakes can only ever start in the range [2, final_square).
            starting_squares[i] = np.random.randint(2, final_square)
            # For snakes 1 <= end_squares[i] < starting_squares[i].
            end_squares[i] = np.random.randint(1, starting_squares[i])
            # Same conditions as before.
            if not np.any(np.unique(starting_squares, return_counts=True)[-1]-1) \
                and starting_squares[i] not in end_squares \
                and end_squares[i] not in starting_squares:
                i += 1

    print(f'starting_squares = {starting_squares}')
            
    print(f'end_squares = {end_squares}')
    
    # Generate snakes_ladders np array of tuples containing (no_ladders + no_snakes) tuples of the 
    # form [start, end].
    snakes_ladders = np.zeros((np.size(starting_squares), 2), dtype=int)
    for i, k in enumerate(starting_squares):
        snakes_ladders[i, :] = [k, end_squares[i]]
        
    print(f'snakes_ladders = {snakes_ladders}')
    return final_square, snakes_ladders
    
def create_square_mean_duration(mean_duration):
    '''
    Create an array of square_mean_duration from mean_duration by adding column of square numbers.

    Parameters
    ----------
    mean_duration : array_like
        Array of size final_square where each element mean_duration[i] gives the average number of 
        turns it takes to reach the final_square from square i.

    Returns
    -------
    square_mean_duration : ndarray
        Array of shape (final_square, 2) of the form [square, mean_duration for that square] going 
        from square 0 (off the board) to square (final_square - 1).

    '''
    
    square_mean_duration = np.zeros((np.size(mean_duration), 2))
    for i, k in enumerate(square_mean_duration):
        k[0] = i
        k[1] = mean_duration[i]
    return square_mean_duration

def calculate_fundamental_matrix(P):
    '''
    Calculate the expected number of turns before being absorbed in an absorbing state of matrix P, 
    t, (equivalent to the average number of turns it takes to reach the final_square, mean_duration)
    and the variation on this, var, using the fundamental matrix, N.

    Parameters
    ----------
    P : array_like
        Transition matrix for an absorbing Markov chain.

    Returns
    -------
    t : ndarray
        The expected number of steps before being absorbed when starting in transient state i.
    var : ndarray
        The variance on the number of steps before being absorbed when starting in transient state 
        i.

    '''
    
    # We use the equation for the fundamental matrix, N:
    # N = (I_t - Q)^-1, where I_t is the identity matrix the same size as Q.
    # (The matrix N gives the expected number of times the chain is in state j, given that 
    # the chain started in state i.)
    
    Q = P[:-1, :-1]
    # NOTE THAT THIS ONLY WORKS FOR A SQUARE MATRIX, ERGO ONLY FOR A SQUARE BOARD!
    # The above may be true for our code, but do not think it is true in general?
    # In fact, it is not true because P is always a square matrix of shape 
    # (final_square+1)x(final_square+1)
    I_t = np.identity(np.size(Q, axis=0))
    # np.linalg.inv(A) gives the inverse matrix of matrix A
    N = np.linalg.inv(I_t - Q)
    
    # The expected number of steps before being absorbed when starting in transient state i is the 
    # ith entry of the vector t = N 1 where 1 is a length-t column vector whose entries are all 1.
    t = np.matmul(N, np.ones((np.shape(N)[1], 1)))
    
    ##########################################################################################
    # NOTE: We do not bother manually setting the mean_duration for \mu when the corresponding 
    # \nu = final_square, because we understand that mean_duration for final_square from the 
    # matrix methods gives 1.
    ##########################################################################################
    
    # The variance on the number of steps before being absorbed when starting in transient state i 
    # is the ith entry of the vector (2N - I_t)t - t_sq, where t_sq is the Hadamard product of t 
    # with itself (i.e. each entry of t is squared).
    # Trivially, t_sq = t**2.
    
    var = np.matmul((2*N - I_t), t) - t**2
    return t, var

def Transfer_Matrices(board_parameters, win_condition):
    
    P = generate_P(*board_parameters, win_condition)
    
    final_square = board_parameters[0]
    # We need a large n to get an accurate value for mean_duration.
    n = final_square*10
     
    # Calculate the probabilites of reaching the final_square in range(n+1) turns.
    p_n_column_list, pi_n_column_list, pi_n_column_list_times_r = calculate_probabilities(
        final_square, n, P=P)
    
    p_n_start_square = p_n_column_list[0, :]
    pi_n_start_square = pi_n_column_list[0, :]
    
    # Plot p_n and pi_n for the starting square as a function of n.
    # Limit the x-axis because n is so large.
    limited_size = int(0.125*np.size(p_n_start_square))
    x = np.linspace(1, limited_size, num=limited_size)
    
    plot_it(x, ys=np.array([p_n_start_square[:limited_size], 
                                   pi_n_start_square[:limited_size]]),  
                    y_labels=np.array(['p_n_start_square', 'pi_n_start_square']),
                    x_name='Number of turns, n', y_name='Probability of victory', 
                    plot_title='Plot of probability of victory against the number of turns played')
    
    # Calculate the average number of turns it takes to reach the final_square from any square.
    mean_duration = calculate_mean_duration(board_parameters, pi_n_column_list_times_r)
    print(f'mean_duration = {mean_duration}')
    
    # We can only get the variance on the mean_duration using the fundamental matrix approach.
    t, var = calculate_fundamental_matrix(P)
    print(f'var = {var}')
    
    # The below code illustrates that mean_duration and t are equivalent (to 8 d.p.).
    # for i, k in enumerate(mean_duration):
    #     if np.around(k, 8) == np.around(t[i], 8):
    #         print('True')
    
    # Order the squares according to the average length of time until victory.
    square_mean_duration = create_square_mean_duration(mean_duration)
    print(f'square_mean_duration = {square_mean_duration}')
    square_mean_duration_ordered = order_square_mean_duration(square_mean_duration)
    print(f'square_mean_duration_ordered = {square_mean_duration_ordered}')
    print(f'\nFor the starting square: \n\n\
mean_duration = {square_mean_duration[0, -1]} \n\
variance = {var[0]}')
    # We add +1 when using np.nonzero and np.argmax (which use indexing), because
    # index = 0 corresponds to n = 1.
    print(f'Minimum game length = {np.nonzero(p_n_start_square)[0][0]+1}')
    print(f'Mode = {np.argmax(pi_n_start_square)+1}')
    print(f'Percentiles: \n[25th, 50th (median), 75th] = ' +
          '{np.searchsorted(p_n_start_square, [0.25, 0.5, 0.75])}')
    
    # Plot mean_duration against square
    plot_it(x=square_mean_duration[:, 0], ys=np.array([square_mean_duration[:, 1]]),
                y_labels=np.array(['square_mean_duration']),
                x_name='Square',
                y_name='Average number of turns until victory', 
                plot_title='Plot of average duration until victory against square',
                individual_points=True)

def Fundamental_Matrix(board_parameters, win_condition):
    P=generate_P(*board_parameters, win_condition)
    t, var = calculate_fundamental_matrix(P)
    print(f't = {t} \nvar = {var}')
    print(f'\nFor the starting square: \n\n\
mean_duration = {t[0]} \n\
variance = {var[0]}')

def method_menu(board_option, board_parameters, win_condition):
    while True:
        print('''
----------------------------------------------
    You are investigating the ''' + board_option + '''
    with the ''' + win_condition + ''' win condition.
----------------------------------------------
Method Menu
-Which method would you like to apply?
            1. Monte Carlo
            2. Transfer Matrices
            3. Fundamental Matrix
            4. Back to Win Condition Menu
                      ''')
        choice = input('Please enter your choice (1-4): ')
        if choice == '1':
            Monte_Carlo(board_parameters, win_condition)
            print(f'\nboard_parameters = {board_parameters}' +
                  f'\nwin_condition = {win_condition}')
        elif choice == '2':
            Transfer_Matrices(board_parameters, win_condition)
            print(f'\nboard_parameters = {board_parameters}' +
                  f'\nwin_condition = {win_condition}')
        elif choice == '3':
            Fundamental_Matrix(board_parameters, win_condition)
            print(f'\nboard_parameters = {board_parameters}' +
                  f'\nwin_condition = {win_condition}')
        elif choice == '4':
            win_condition_menu(board_option, board_parameters)
        else:
            print('Invalid input. Please enter a digit between 1 and 4.')
            
def win_condition_menu(board_option, board_parameters):
    while True:
        print('''
----------------------------------------------
    You are investigating the ''' + board_option + '''.
----------------------------------------------
Win Condition Menu
-Which win condition would you like to apply?
            1. Over 
            - (need to reach final_square or beyond)
            2. Exact
            - (stay put if roll past final_square)
            3. Bounce
            - (bounce back if roll past final_square)
            4. Back to Board Menu
                      ''')
        choice = input('Please enter your choice (1-4): ')
        if choice == '1':
            method_menu(board_option, board_parameters, 'Over')
        elif choice == '2':
            method_menu(board_option, board_parameters, 'Exact')
        elif choice == '3':
            method_menu(board_option, board_parameters, 'Bounce')
        elif choice == '4':
            board_menu()
        else:
            print('Invalid input. Please enter a digit between 1 and 4.')
            
def board_menu():
    while True:
        print('''
**********************************************
---------------SNAKES & LADDERS---------------
**********************************************
Board Menu
-What board would you like to investigate?
            1. Empty Board
            2. Example Board
            3. Random Board
            4. Quit
                      ''')
        choice = input('Please enter your choice (1-4): ')
        if choice == '1':
            # board_parameters are a numpy array containing [final_square, snakes_ladders]
            board_parameters = np.array([100, np.empty([0,0])])
            win_condition_menu('Empty Board', board_parameters)
        elif choice == '2':
            snakes_ladders = np.array([[1, 38], [4, 14], [9, 31], [16, 6], [21, 42], [28, 84], 
                                           [36, 44], [47, 26], [49, 11], [51, 67], [56, 53], 
                                           [62, 19], [64, 60], [71, 91], [80, 100], [87, 24], 
                                           [93, 73], [95, 75], [98, 78]])
            board_parameters = np.array([100, snakes_ladders])
            win_condition_menu('Example Board', board_parameters)
        elif choice == '3':
            board_parameters = generate_random_board_do_not_allow_same_start_end()
            win_condition_menu('Random Board', board_parameters)
        elif choice == '4':
            print('Goodbye!')
            sys.exit()
        else:
            print('Invalid input. Please enter a digit between 1 and 4.')
            
def get_number_of_simulations():
    while True:
        try:
            simulations = int(input('Please enter the number of simulations you would ' +
                                    'like to run \nthe Monte Carlo method for ' + 
                                    'as an integer (100-100000): '))
        except ValueError:
            print('You did not enter an integer.')
            continue
        if 100 <= simulations <= 100000:
            return simulations
        else:
            print('Invalid input. Please enter a number between 100 and 100000.')
            
def main():
    board_menu()
    
if __name__ == '__main__':
    main()