import numpy as np
import spookyhash

# Super basic prototype to illustrate the idea

def update_bitvector(array, bit, axis=None, value=1):
	# Update the specified bit of the bitvector in array with the value.
	# Supports collections of bitvectors via the container_axis argument.
	# Does not do bounds checking or shape checks for speed reasons.
	# 
	# array: (N1 x N2 x N3 ... Nm x bitvector_container_size) array. Must be
	#     a little-endian integer type and the bitvectors must be the last dimension.
	# bit: Integer in range [0 ... bitvector_container_size-1].
	# container_axis: If array has m+1 dimensions, axis is a tuple of m indices
	#     to select the correct bitvector to update. Can be called with an integer.
	# value: Value to use to update the bitvector location. Either 0 or 1.
	# Returns: array with the specified bit in the specified bitvector set.
	backing_type = array.dtype
	vars_per_chunk = backing_type.itemsize * 8
	chunk_id = variable_id // vars_per_chunk
	num_left_shifts = variable_id % vars_per_chunk  # bit position for little-endian types
	chunk = np.array([1], dtype=backing_type)
	chunk = np.left_shift(chunk, num_left_shifts)
	# Construct the index tuple to update in array.
	if container_axis is not None:
		try:
			update_index = tuple(list(container_axis) + [chunk_id])
		except TypeError:
			update_index = tuple([container_axis] + [chunk_id])
	else:
		update_index = chunk_id
	# Update with either 0 or 1.
	if value:
		array[update_index] = np.bitwise_or(array[update_index], chunk)
	else:
		chunk = np.bitwise_not(chunk)
		array[update_index] = np.bitwise_and(array[update_index], chunk)
	return array

def countsort_variable_ids(variable_weight, num_variables, num_equations):
	sorted_variable_ids = list(range(num_variables))
	counts = np.zeros(size=num_equations+1, dtype=int)
	for variable_id in range(num_variables):
		counts[variable_weight[variable_id]] += 1		
	counts = np.cumsum(counts)
	for variable_id in reversed(range(num_variables)):
		count_idx = variable_weight[variable_id]
		counts[count_idx] -= 1
		sorted_variable_ids[counts[count_idx]] = variable_id
	return sorted_variable_ids


# def gaussian_elimination():

# Java method:
# 	public boolean gaussianElimination(final long[] solution) {
# 		assert solution.length == numVars;
# 		for (final Modulo2Equation equation: equations) equation.updateFirstVar();

# 		if (! echelonForm()) return false;

# 		for (int i = equations.size(); i-- != 0;) {
# 			final Modulo2Equation equation = equations.get(i);
# 			if (equation.isIdentity()) continue;
# 			assert solution[equation.firstVar] == 0 : equation.firstVar;
# 			solution[equation.firstVar] = equation.c ^ Modulo2Equation.scalarProduct(equation.bits, solution);
# 		}

# 		return true;
# 	}


def lazy_gaussian_elimination(var_to_equation, y):
	# Performs lazy gaussian elimination on the linear system. We consider
	# a system of N equations and M variables, where the equations and variables
	# are identified with integer labels ranging from [0 ... N-1] and [0 ... M-1].
	# var_to_equation: a list or dict that maps variable_ids to lists of equation_ids
	# y: The desired output of the system (1 x N integer vector) of 1s and 0s.

	num_variables = len(var_to_equation)  # number of variables
	num_equations = len(y)  # number of equations / output values

	# Build the equations vector. The equations are represented as a set of bitvectors.
	# Each bitvector is packed into a container of unsigned integers, which we store as a 2D array.
	backing_type = np.dtype('>i8')
	vars_per_chunk = backing_type.itemsize * 8
	equation_container_size = num_variables / vars_per_chunk
	equation_container_size = int(np.ceil(equation_container_size))
	# Binary equations, where '1' at index (i, j) indicates a '1' for variable i in equation j.
	equations = np.zeros([num_equations, equation_container_size], dtype=backing_type)

	variable_weight = np.zeros(size=num_variables)  # Number of sparse equations containing variable_id.
	equation_priority = np.zeros(size=num_equations)  # Number of idle variables in equation_id.

	for variable_id in range(num_variables):
		# This set() transformation is required to avoid incrementing the weight / priority
		# for situations where variable_id appears in the same equation multiple times.
		# (I'm not sure how reasonable it is for this to occur but the Java lib does it).
		unique_equations = set(var_to_equation[variable_id])
		for equation_id in var_to_equation[variable_id]:
			equations = update_bitvector(equations, variable_id, axis=equation_id)
			variable_weight[variable_id] += 1
			equation_priority[equation_id] += 1
			# Again, I'm not sure whether this is necessary if called with non-dumb inputs.
			var_to_equation[variable_id] = unique_equations

	# Sort variable ids by weight.
	sorted_variable_ids = countsort_variable_ids(variable_weight, num_variables, num_equations)

	# Sparse equations with weight 0 or 1. This possibly needs a re-name.
	sparse_equation_ids = []
	# Dense equations with entirely active variables.
	dense_equation_ids = []
	# Equations that define a solved variable in terms of active variables.
	solved_equation_ids = []
	solved_variable_ids = []
	# List of currently-idle variables. Starts as a bit vector of all 1's.
	# We fill in 0s as variables become non-idle.
	idle_variable_indicator = -1 * np.ones([num_equations, equation_container_size], dtype=backing_type)

	num_active_variables = 0
	num_remaining_equations = num_equations

	while (num_remaining_equations >= 0):
		if not sparse_equation_ids:
			# If there are no sparse equations with priority 0 or 1,
			# we make another variable active and see if this changes.
			# Skip variables with weight = 0 (these are already solved).
			variable_id = sorted_variable_ids.pop()
			while not variable_weight[variable_id]:
				variable_id = sorted_variable_ids.pop()
			# Mark variable as no longer idle.
			idle_variable_indicator = update_bitvector(idle_variable_indicator, variable_id, value=0)
			num_active_variables += 1
			print(f"Making variable {variable_id:d} with weight {weight[variable_id]:d} active ({num_remaining_equations:d} equations remaining).")
			for equation_id in var_to_equation[variable_id]:
				equation_priority[equation_id] -= 1
				if equation_priority == 1:
					sparse_equation_ids.append(equation_id)
		else:
			num_remaining_equations -= 1
			equation_id = sparse_equation_ids.pop()
			print(f"Equation {equation_id:d} with priority {equation_priority[equation_id]:d}.")
			if equation_priority[equation_id] == 0:
				equation_is_empty = np.sum(equations[equation_id, :])
				if not equation_is_empty:
					# This equation needs to be solved by standard Gaussian elimination.
					# Since the priority is 0, all of the variables are active.
					dense_equation_ids.append(equation_id)
				elif y[equation_id] != 0:  # The equation is unsolvable.
					return None
				# The remaining case corresponds to an identity equation (empty, but so is the output).
			elif equation_priority[equation_id] == 1:
				# If there is only 1 idle variable, then equation_id is solved.
				# First, we find the pivot (the id of the only idle variable left in equation_id)
				# This is the location where both the equation and idle variable indicator are 1.
				for chunk in range(equation_container_size):
					chunk_contents = np.bitwise_and(equation[equation_id, chunk], idle_variable_indicator[chunk])
					if chunk_contents > 0:
						break
				# Note: chunk_contents is a power of 2, so the log2 can be done via bit shifting.
				pivot_id = chunk * vars_per_chunk + int(np.log2(chunk_contents))
				solved_variable_ids.append(pivot_id)
				solved_equation_ids.append(equation_id)
				# By making the weight 0, we will skip pivot_id when looking for new active variables.
				variable_weight[pivot_id] = 0
				# Remove the pivot from all the other equations and update priorities.
				for other_equation_id in var_to_equation[pivot_id]:
					if other_equation_id != equation_id:
						equation_priority[other_equation_id] -= 1
						if equation_priority[other_equation_id] == 1:
							sparse_equation_ids.append(other_equation_id)
						print(f"Adding equation {equation_id:d} to equation {other_equation_id:d}.")
						# Perform a step of Gaussian elimination.
						equations[other_equation_id,:] = np.bitwise_xor(equations[other_equation_id,:], equations[equation_id,:])
						# Note: The new equation may contain new non-pivot variables, but the priority does not need to change
						# because the priority is the number of idle variables (and the non-pivots are not idle). Likewise,
						# the variable weights don't need to change either (we only use weights of idle variables).

	print(f"{num_active_variables:d} active of {num_variables:d} total variables ({num_active_variables / num_variables:.2f}%).")

	print(f"Dense equations: {dense_equation_ids}")
	print(f"Solved equations: {solved_equation_ids}")
	print(f"Solved variables: {solved_variable_ids}")

	# Now do full Gaussian elimination on the remaining equations.

	# Java method:
	# final Modulo2System denseSystem = new Modulo2System(numVars, dense);
	# if (! denseSystem.gaussianElimination(solution)) return false;  // numVars >= denseSystem.numVars

	# if (DEBUG) System.err.println("Solution (dense): " + Arrays.toString(solution));

	# for (int i = solved.size(); i-- != 0;) {
	# 	final Modulo2Equation equation = solved.get(i);
	# 	final int pivot = pivots.getInt(i);
	# 	assert solution[pivot] == 0 : pivot;
	# 	solution[pivot] = equation.c ^ Modulo2Equation.scalarProduct(equation.bits, solution);

	# if (DEBUG) System.err.println("Solution (all): " + Arrays.toString(solution));

	return true;


# def peel_hypergraph():
# Java method: (to be found - think it's in GV3CompressedFunction)	





def construct_csf(keys, values):
	"""
	Constructs a compressed static function. This implementation sacrifices a lot of the 
	configurability for the sake of simplicity.

	Arguments:
		keys: An iterable collection of N unique hashable items.
		values: An iterable collection of N values. 

	Returns:
		A csf structure supporting the .query(key) method.
	"""

	# 1. Divide keys into buckets


	# 2. 


	return



if __name__ == '__main__':
	num_equations = 3
	backing_array_size = 2
	backing_type = np.dtype('<i4')
	array = np.zeros([num_equations, backing_array_size], dtype = backing_type)
	
	array = update_bitvector(array, 0, 0)
	array = update_bitvector(array, 2, 0)
	array = update_bitvector(array, 63, 0)
	array = update_bitvector(array, 1, 1)

	with open('tst.bin', 'wb') as f:
		f.write(array.tobytes())

