/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2016 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Authors: Wolfgang Bangerth, 1999,
 *          Guido Kanschat, 2011
 */


#include <deal.II/base/work_stream.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <fstream>
#include <iostream>

using namespace dealii;

int REFINEMENT_LEVEL;

class Step3
{
public:
  Step3 ();

  void run ();

void assemble_system_1cell(const auto& cell,auto& scratch,auto& data);
void copy_1cell(const auto& data);

private:
	void refine_grid();
  void make_grid ();
  void setup_system ();
  void assemble_system ();
  void solve ();
  void output_results (int) const;

  Triangulation<2>     triangulation;
  FE_Q<2>              fe;
  DoFHandler<2>        dof_handler;
	
  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double>       solution;
  Vector<double>       system_rhs;
	
	
	AffineConstraints<double> constraints;
};


Step3::Step3 ()
  :
  fe (1),
  dof_handler (triangulation)
{}



void Step3::make_grid ()
{
  GridGenerator::hyper_cube (triangulation, -1, 1);
  triangulation.refine_global (REFINEMENT_LEVEL);

//  std::cout << "Number of active cells: "
//            << triangulation.n_active_cells()
//            << std::endl;
}




void Step3::setup_system ()
{
  dof_handler.distribute_dofs (fe);
//  std::cout << "Number of degrees of freedom: "
//            << dof_handler.n_dofs()
//            << std::endl;

	constraints.clear();
	DoFTools::make_hanging_node_constraints(dof_handler,constraints);

	VectorTools::interpolate_boundary_values(
		dof_handler,
		0,
		Functions::ZeroFunction<2>(),
		constraints);

	constraints.close();
	DynamicSparsityPattern dsp(dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern (dof_handler, dsp,constraints,false);
	
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit (sparsity_pattern);

  solution.reinit (dof_handler.n_dofs());
  system_rhs.reinit (dof_handler.n_dofs());
}

const double PI=acos(-1.);

double fun(const Point<2>& p){
	return 40*PI*PI*sin(2*PI*p[0])*sin(6*PI*p[1]);
}


class u_fun_class : public Function<2> 
{
	public:
	
	u_fun_class() : Function<2>() {}
	
	virtual double value(const Point<2>& p, const unsigned int comp=0)
		const override 
	{
		return sin(2*PI*p[0])*sin(6*PI*p[1]);
	}
} u_fun ;

struct ScratchData {
	FEValues<2> fe_values ;
	
	ScratchData(
		const FiniteElement<2>& fe,
		const Quadrature<2>& q,
		const UpdateFlags flags):
			fe_values(fe,q,flags){}
			
	ScratchData(const ScratchData& rhs):
		fe_values(
			rhs.fe_values.get_fe(),
			rhs.fe_values.get_quadrature(),
			rhs.fe_values.get_update_flags()){}
};

struct PerTaskData {

	FullMatrix<double>   cell_matrix;
	Vector<double>       cell_rhs;
	std::vector<types::global_dof_index> local_dof_indices;
	
	
	PerTaskData(const FiniteElement<2>& fe):
		cell_matrix(fe.dofs_per_cell,fe.dofs_per_cell),
		cell_rhs(fe.dofs_per_cell),
		local_dof_indices(fe.dofs_per_cell){}
};

void Step3::assemble_system_1cell(const auto& cell,auto& scratch,auto& data){
		
	const unsigned int n_q_points = scratch.fe_values.n_quadrature_points;
	const unsigned int dofs_per_cell = scratch.fe_values.dofs_per_cell;
	scratch.fe_values.reinit (cell);

	data.cell_matrix = 0;
	data.cell_rhs = 0;

	for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
	{
		for (unsigned int i=0; i<dofs_per_cell; ++i)
		for (unsigned int j=0; j<dofs_per_cell; ++j)
			data.cell_matrix(i,j) += (
				scratch.fe_values.shape_grad (i, q_index) *
				scratch.fe_values.shape_grad (j, q_index) *
				scratch.fe_values.JxW (q_index));


		for (unsigned int i=0; i<dofs_per_cell; ++i)
			data.cell_rhs(i) += (
				scratch.fe_values.shape_value (i, q_index) *
				fun( scratch.fe_values.quadrature_point(q_index)  )*
				scratch.fe_values.JxW (q_index));
	}
	
	cell->get_dof_indices ( data.local_dof_indices);
	
}


void Step3::copy_1cell(const auto& data){
	constraints.distribute_local_to_global(
		data.cell_matrix,
		data.cell_rhs,
		data.local_dof_indices,
		system_matrix,
		system_rhs);	
}

void Step3::assemble_system ()
{
	QGauss<2>  quadrature_formula(2);
	
	ScratchData scratch(fe,quadrature_formula,
		update_values | update_gradients 
		| update_JxW_values | update_quadrature_points );
	
	PerTaskData data(fe);

	WorkStream::run(
		dof_handler.begin_active(),
		dof_handler.end(),
		*this,
		&Step3::assemble_system_1cell,
		&Step3::copy_1cell,
		scratch,
		data);
	
/*
	for (auto cell: dof_handler.active_cell_iterators())
	{
		
		assemble_system_1cell(cell,scratch,data);
		copy_1cell(data);
		
	}
*/
	std::map<types::global_dof_index,double> boundary_values;
	VectorTools::interpolate_boundary_values (dof_handler,
											0,
											ZeroFunction<2>(),
											boundary_values);
	MatrixTools::apply_boundary_values (boundary_values,
									  system_matrix,
									  solution,
									  system_rhs);
}



void Step3::solve ()
{
  SolverControl           solver_control (1000, 1e-12);
  SolverCG<>              solver (solver_control);

  solver.solve (system_matrix, solution, system_rhs,
                PreconditionIdentity());
				
	constraints.distribute(solution);
}



void Step3::output_results (int i) const
{
  DataOut<2> data_out;
  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution, "solution");
  data_out.build_patches ();

  std::ofstream output("solution"+std::to_string(i)+".svg");
  data_out.write_svg (output);
  
  Vector<double> exact_solution(dof_handler.n_dofs());
  VectorTools::interpolate(dof_handler,u_fun,exact_solution);
  
  exact_solution -= solution;
  std::cout << exact_solution.linfty_norm() << std::endl; // difference between exact and FE solutions
  
}

void Step3::refine_grid(){
	
	Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
	KellyErrorEstimator<2>::estimate(
		dof_handler,
		QGauss<1>(2),
		std::map<types::boundary_id, const Function<2> *>(),
		solution,
		estimated_error_per_cell);

	GridRefinement::refine_and_coarsen_fixed_number(triangulation,
		estimated_error_per_cell,
		0.3,
		0.03);
		
	triangulation.execute_coarsening_and_refinement();
}

void Step3::run ()
{

  	make_grid ();
	for(int i=0;i<=6;++i){
		
		if(i)refine_grid();
		setup_system ();
		assemble_system ();
		solve ();
		output_results (i);	
	}
}



int main (int narg, char** args)
{

	std::cout<<"threads: "<<MultithreadInfo::n_threads()<<std::endl;

	REFINEMENT_LEVEL=3;

  //deallog.depth_console (2);

  Step3 laplace_problem;
  laplace_problem.run ();

  return 0;
}
