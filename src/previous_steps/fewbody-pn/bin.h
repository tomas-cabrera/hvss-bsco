/* -*- linux-c -*- */
/* binbin.h

   Copyright (C) 2002-2004 John M. Fregeau
   
   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.
   
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#define FB_TIDALTOL 1.0e-5
#define FB_SPEEDTOL 5.0e-3

#define FB_PN1 1
#define FB_PN2 1
#define FB_PN25 1
#define FB_PN3 0
#define FB_PN35 0
#define FB_QUAD 0

#define FB_M00 (0.5 * FB_CONST_MSUN)
#define FB_M01 (0.5 * FB_CONST_MSUN)

#define FB_CHI00 -1.
#define FB_CHI01 -1.

#define FB_R00 (0.0 * FB_CONST_RSUN)
#define FB_R01 (0.0 * FB_CONST_RSUN)

#define FB_A0 (0.000001 * FB_CONST_AU)

#define FB_E0 0.0

#define FB_DT 1.0 /* approximate output dt */
#define FB_TSTOP 1.0e6 /* in units of t_dyn */
#define FB_TCPUSTOP 3600.0 /* in seconds */

#define FB_ABSACC 1.0e-9 /* absolute accuracy of integrator */
#define FB_RELACC 1.0e-9 /* relative accuracy of integrator */
#define FB_NCOUNT 500 /* number of timesteps between calls to classify() */

#define FB_KS 0

#define FB_FEXP 3.0 /* expansion factor of merger product */

#define FB_SEED 0UL
#define FB_DEBUG 0

void print_usage(FILE *stream);
int calc_units(fb_obj_t *obj[2], fb_units_t *units);
