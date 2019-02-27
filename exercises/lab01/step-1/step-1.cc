/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2015 by the deal.II authors
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

 */



#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>

#include <iostream>
#include <fstream>
#include <cmath>

using namespace dealii;

template<int DIM>
void info(const Triangulation<DIM>& T){
	int nac=0,nc=0;
	for(auto cell: T) nc++;
	for(auto cell: T.active_cell_iterators()) nac++;
	std::cout<<"Active cells: "<<nac<<"\n";
	std::cout<<"Total cells: "<<nc<<"\n";
	std::cout<<"Levels: "<<T.n_levels()<<"\n";
}


void first_grid ()
{
  Triangulation<2> triangulation;

  GridGenerator::hyper_cube (triangulation);
  triangulation.refine_global (4);

  std::ofstream out ("grid-1.eps");
  GridOut grid_out;
  grid_out.write_eps (triangulation, out);
  std::cout << "Grid written to grid-1.eps" << std::endl;
  info(triangulation);
}




void second_grid ()
{
  Triangulation<2> triangulation;

  const Point<2> center (1,0);
  const double inner_radius = 0.5,
               outer_radius = 1.0;
  GridGenerator::hyper_shell (triangulation,
                              center, inner_radius, outer_radius,
                              10);
  triangulation.set_all_manifold_ids(0);
  const SphericalManifold<2> manifold_description(center);
//  triangulation.set_manifold (0, manifold_description);

  for (unsigned int step=0; step<5; ++step)
    {
      Triangulation<2>::active_cell_iterator
      cell = triangulation.begin_active(),
      endc = triangulation.end();
      for (; cell!=endc; ++cell)
        {
          for (unsigned int v=0;
               v < GeometryInfo<2>::vertices_per_cell;
               ++v)
            {
              const double distance_from_center
                = center.distance (cell->vertex(v));

              if (std::fabs(distance_from_center - inner_radius) < 1e-10)
                {
                  cell->set_refine_flag ();
                  break;
                }
            }
        }

      triangulation.execute_coarsening_and_refinement ();
    }


  std::ofstream out ("grid-2.eps");
  GridOut grid_out;
  grid_out.write_eps (triangulation, out);
  out.close();
  out.open("grid-2.svg");
  grid_out.write_svg (triangulation, out);

  std::cout << "Grid written to grid-2.eps" << std::endl;
	
	info(triangulation);
//  triangulation.set_manifold (0);
}


void third_grid(){
	Triangulation<2> T;
	Point<2> center(0,0);
	GridGenerator::hyper_ball(T,center);
	const SphericalManifold<2> man(center);
	
	std::ofstream out ("grid-3.svg");
	GridOut grid_out;
	grid_out.write_svg (T, out);
}


int main ()
{
  first_grid ();
  second_grid ();
  third_grid();
  std::cout<<"Exited gracefully\n";
  return 0;
}
