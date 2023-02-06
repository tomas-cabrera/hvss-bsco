/* -*- linux-c -*- */
/* binsingle.c

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

#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <getopt.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_rng.h>
#include "fewbody.h"
#include "singlesingle.h"

/* print the usage */
void print_usage(FILE *stream)
{
	fprintf(stream, "USAGE:\n");
	fprintf(stream, "  binsingle [options...]\n");
	fprintf(stream, "\n");
	fprintf(stream, "OPTIONS:\n");
	fprintf(stream, "  -m --m0 <m0/MSUN>            : set mass of first star [%.6g]\n", FB_M0/FB_CONST_MSUN);
	fprintf(stream, "  -n --m1 <m1/MSUN>            : set mass of second star [%.6g]\n", FB_M1/FB_CONST_MSUN);
	fprintf(stream, "  -r --r0 <r0/RSUN>            : set radius of first star [%.6g]\n", FB_R0/FB_CONST_RSUN);
	fprintf(stream, "  -g --r1 <r1/RSUN>            : set radius of second star [%.6g]\n", FB_R1/FB_CONST_RSUN);
	fprintf(stream, "  -v --vinf <vinf/KM/S>        : set velocity at infinity [%.6g]\n", FB_VINF);
	fprintf(stream, "  -b --b <b/AU>                : set impact parameter [%.6g]\n", FB_B);
	fprintf(stream, "  -t --tstop <tstop/t_dyn>     : set stopping time [%.6g]\n", FB_TSTOP);
	fprintf(stream, "  -D --dt <dt/t_dyn>           : set approximate output dt [%.6g]\n", FB_DT);
	fprintf(stream, "  -c --tcpustop <tcpustop/sec> : set cpu stopping time [%.6g]\n", FB_TCPUSTOP);
	fprintf(stream, "  -A --absacc <absacc>         : set integrator's absolute accuracy [%.6g]\n", FB_ABSACC);
	fprintf(stream, "  -R --relacc <relacc>         : set integrator's relative accuracy [%.6g]\n", FB_RELACC);
	fprintf(stream, "  -N --ncount <ncount>         : set number of integration steps between calls\n");
	fprintf(stream, "                                 to fb_classify() [%d]\n", FB_NCOUNT);
	fprintf(stream, "  -z --tidaltol <tidaltol>     : set tidal tolerance [%.6g]\n", FB_TIDALTOL);
	fprintf(stream, "  -y --speedtol <speedtol>     : set speed tolerance [%.6g]\n", FB_SPEEDTOL);
	fprintf(stream, "  -x --fexp <f_exp>            : set expansion factor of merger product [%.6g]\n", FB_FEXP);
	fprintf(stream, "  -P --PN1 <PN1>               : PN1 terms on? [%d]\n", FB_PN1);
	fprintf(stream, "  -Q --PN2 <PN2>               : PN2 terms on? [%d]\n", FB_PN2);
	fprintf(stream, "  -S --PN25 <PN25>             : PN2.5 terms on? [%d]\n", FB_PN25);
	fprintf(stream, "  -T --PN3 <PN3>               : PN3 terms on? [%d]\n", FB_PN3);
	fprintf(stream, "  -U --PN35 <PN35>             : PN3.5 terms on? [%d]\n", FB_PN35);
	fprintf(stream, "  -k --ks                      : turn K-S regularization on or off [%d]\n", FB_KS);
	fprintf(stream, "  -s --seed                    : set random seed [%ld]\n", FB_SEED);
	fprintf(stream, "  -d --debug                   : turn on debugging\n");
	fprintf(stream, "  -W --waves                   : prints the second-time dervative of the mass quadrupole moment [%d]\n",FB_QUAD);
	fprintf(stream, "  -V --version                 : print version info\n");
	fprintf(stream, "  -h --help                    : display this help text\n");
}

/* calculate the units used */
int calc_units(double b, double vinf, fb_units_t *units)
{
	units->v = vinf; 
	units->l = b;
	//units->m = M;
	//units->l = units->m / fb_sqr(units->v) * FB_CONST_G;
	units->t = units->l / units->v;
	units->m = units->l * fb_sqr(units->v) / FB_CONST_G;
	units->E = units->m * fb_sqr(units->v);
	
	return(0);
}

/* the main attraction */
int main(int argc, char *argv[])
{
	int i, j, k;
	unsigned long int seed;
	double m0, m1, r0, r1;
	double rtid, vinf, b, M, mu, Ei, E, Lint[3], Li[3], l0[3], l1[3], L[3], r[3], t;
	double chi;
	int bid,sid;
	fb_hier_t hier;
	fb_input_t input;
	fb_ret_t retval;
	fb_units_t units;
	char string1[FB_MAX_STRING_LENGTH], string2[FB_MAX_STRING_LENGTH];
	gsl_rng *rng;
	const gsl_rng_type *rng_type=gsl_rng_mt19937;
	const char *short_opts = "m:n:r:g:C:v:b:t:D:c:A:R:N:z:y:x:P:Q:S:T:U:k:s:dVh";
	const struct option long_opts[] = {
		{"m0", required_argument, NULL, 'm'},
		{"m1", required_argument, NULL, 'n'},
		{"r0", required_argument, NULL, 'r'},
		{"r1", required_argument, NULL, 'g'},
		{"chi", required_argument, NULL, 'C'},
		{"vinf", required_argument, NULL, 'v'},
		{"b", required_argument, NULL, 'b'},
		{"tstop", required_argument, NULL, 't'},
		{"dt", required_argument, NULL, 'D'},
		{"tcpustop", required_argument, NULL, 'c'},
		{"absacc", required_argument, NULL, 'A'},
		{"relacc", required_argument, NULL, 'R'},
		{"ncount", required_argument, NULL, 'N'},
		{"tidaltol", required_argument, NULL, 'z'},
		{"speedtol", required_argument, NULL, 'y'},
		{"fexp", required_argument, NULL, 'x'},
		{"PN1", required_argument, NULL, 'P'},
		{"PN2", required_argument, NULL, 'Q'},
		{"PN25", required_argument, NULL, 'S'},
		{"PN3", required_argument, NULL, 'T'},
		{"PN35", required_argument, NULL, 'U'},
		{"ks", required_argument, NULL, 'k'},
		{"seed", required_argument, NULL, 's'},
		{"debug", no_argument, NULL, 'd'},
		{"waves", no_argument, NULL, 'W'},
		{"version", no_argument, NULL, 'V'},
		{"help", no_argument, NULL, 'h'},
		{NULL, 0, NULL, 0}
	};

	/* set parameters to default values */
	m0 = FB_M0;
	m1 = FB_M1;
	r0 = FB_R0;
	r1 = FB_R1;
	vinf = FB_VINF;
	b = FB_B;
	chi = -1.;
	input.ks = FB_KS;
	input.tstop = FB_TSTOP;
	input.Dflag = 0;
	input.dt = FB_DT;
	input.tcpustop = FB_TCPUSTOP;
	input.absacc = FB_ABSACC;
	input.relacc = FB_RELACC;
	input.ncount = FB_NCOUNT;
	input.tidaltol = FB_TIDALTOL;
	input.speedtol = FB_SPEEDTOL;
	input.fexp = FB_FEXP;
	input.PN1 = FB_PN1;
	input.PN2 = FB_PN2;
	input.PN25 = FB_PN25;
	input.PN3 = FB_PN3;
	input.PN35 = FB_PN35;
	input.fb_quad = FB_QUAD;
	seed = FB_SEED;
	fb_debug = FB_DEBUG;

	suppression=0.;
	J_pn=0;
	E_pn=0;
	e_pn=0;
	v_kick=0;
 	e_newt=0;
	
	while ((i = getopt_long(argc, argv, short_opts, long_opts, NULL)) != -1) {
		switch (i) {
		case 'm':
			m0 = atof(optarg) * FB_CONST_MSUN;
			break;
		case 'n':
			m1 = atof(optarg) * FB_CONST_MSUN;
			break;
		case 'r':
			r0 = atof(optarg) * FB_CONST_RSUN;
			break;
		case 'g':
			r1 = atof(optarg) * FB_CONST_RSUN;
			break;
		case 'v':
			vinf = atof(optarg) * 1e5;
			if (vinf < 0.0) {
				fprintf(stderr, "vinf must be non-negative\n");
				return(1);
			}
			break;
		case 'b':
			b = atof(optarg) * FB_CONST_AU;
			if (b < 0.0) {
				fprintf(stderr, "b must be non-negative\n");
				return(1);
			}
			break;
		case 't':
			input.tstop = atof(optarg);
			break;
		case 'D':
			input.Dflag = 1;
			input.dt = atof(optarg);
			if (input.dt < 0) input.Dflag = 2;
			break;
		case 'c':
			input.tcpustop = atof(optarg);
			break;
		case 'C':
			chi = atof(optarg);
			break;
		case 'A':
			input.absacc = atof(optarg);
			break;
		case 'R':
			input.relacc = atof(optarg);
			break;
		case 'N':
			input.ncount = atoi(optarg);
			break;
		case 'z':
			input.tidaltol = atof(optarg);
			break;
		case 'y':
			input.speedtol = atof(optarg);
			break;
		case 'x':
			input.fexp = atof(optarg);
			break;
		case 'P':
			input.PN1 = atoi(optarg);
			break;
		case 'Q':
			input.PN2 = atoi(optarg);
			break;
		case 'S':
			input.PN25 = atoi(optarg);
			break;
		case 'T':
			input.PN3 = atoi(optarg);
			break;
		case 'U':
			input.PN35 = atoi(optarg);
			break;
		case 'k':
			input.ks = atoi(optarg);
			break;
		case 's':
			seed = atol(optarg);
			break;
		case 'W':
			input.fb_quad = 1;
			break;
		case 'd':
			fb_debug = 1;
			break;
		case 'V':
			fb_print_version(stdout);
			return(0);
		case 'h':
			fb_print_version(stdout);
			fprintf(stdout, "\n");
			print_usage(stdout);
			return(0);
		default:
			break;
		}
	}
	
	/* check to make sure there was nothing crazy on the command line */
	if (optind < argc) {
		print_usage(stdout);
		return(1);
	}

	/* initialize a few things for integrator */
	t = 0.0;
	hier.nstarinit = 2;
	hier.nstar = 2;
	fb_malloc_hier(&hier);
	fb_init_hier(&hier);

	/* put stuff in log entry */
	snprintf(input.firstlogentry, FB_MAX_LOGENTRY_LENGTH, "  command line:");
	for (i=0; i<argc; i++) {
		snprintf(&(input.firstlogentry[strlen(input.firstlogentry)]), 
			 FB_MAX_LOGENTRY_LENGTH-strlen(input.firstlogentry), " %s", argv[i]);
	}
	snprintf(&(input.firstlogentry[strlen(input.firstlogentry)]), 
		 FB_MAX_LOGENTRY_LENGTH-strlen(input.firstlogentry), "\n");
	
	/* print out values of paramaters */
	fprintf(stderr, "PARAMETERS:\n");
	fprintf(stderr, "  ks=%d  seed=%ld\n", input.ks, seed);
	fprintf(stderr, "  m0=%.6g MSUN  m1=%.6g MSUN  r0=%.6g RSUN  r1=%.6g RSUN\n", \
		m0/FB_CONST_MSUN, m1/FB_CONST_MSUN, r0/FB_CONST_RSUN, r1/FB_CONST_RSUN);
	fprintf(stderr, "  vinf=%.6g KM/S  b=%.6g AU  tstop=%.6g  tcpustop=%.6g\n", \
		vinf/1e5, b/FB_CONST_AU, input.tstop, input.tcpustop);
	fprintf(stderr, "  tidaltol=%.6g  speedtol=%.6g  abs_acc=%.6g  rel_acc=%.6g  ncount=%d  fexp=%.6g\n", \
		input.tidaltol, input.speedtol, input.absacc, input.relacc, input.ncount, input.fexp);
	fprintf(stderr, "  PN1=%d  PN2=%d  PN25=%d  PN3=%d  PN35=%d\n\n", \
		input.PN1, input.PN2, input.PN25, input.PN3, input.PN35);

	/* initialize GSL rng */
	gsl_rng_env_setup();
	rng = gsl_rng_alloc(rng_type);
	gsl_rng_set(rng, seed);

	/* give the objects some properties */
	for (j=0; j<hier.nstar; j++) {
		hier.hier[hier.hi[1]+j].ncoll = 1;
		hier.hier[hier.hi[1]+j].id[0] = j;
		snprintf(hier.hier[hier.hi[1]+j].idstring, FB_MAX_STRING_LENGTH, "%d", j);
		hier.hier[hier.hi[1]+j].n = 1;
		hier.hier[hier.hi[1]+j].obj[0] = NULL;
		hier.hier[hier.hi[1]+j].obj[1] = NULL;
		hier.hier[hier.hi[1]+j].Eint = 0.0;
		hier.hier[hier.hi[1]+j].Lint[0] = 0.0;
		hier.hier[hier.hi[1]+j].Lint[1] = 0.0;
		hier.hier[hier.hi[1]+j].Lint[2] = 0.0;
	}

/*	
	hier.hier[hier.hi[1]+0].R = r0;
	hier.hier[hier.hi[1]+1].R = r10;
	hier.hier[hier.hi[1]+2].R = r11;
*/

	hier.hier[hier.hi[1]+0].R = 2 * FB_REFF_BH * FB_CONST_G * m0 / (FB_CONST_C*FB_CONST_C);
	hier.hier[hier.hi[1]+1].R = 2 * FB_REFF_BH * FB_CONST_G * m1 / (FB_CONST_C*FB_CONST_C);


	hier.hier[hier.hi[1]+0].m = m0;
	hier.hier[hier.hi[1]+1].m = m1;

	hier.hier[hier.hi[1]+0].chi = chi;
	hier.hier[hier.hi[1]+1].chi = chi;

	hier.obj[0] = &(hier.hier[hier.hi[1]+0]);
	hier.obj[1] = &(hier.hier[hier.hi[1]+1]);

	/* get the units and normalize */
	calc_units(b, vinf, &units);
	fb_normalize(&hier, units);

	fprintf(stderr, "UNITS:\n");
	fprintf(stderr, "  v=v_crit=%.6g km/s  l=%.6g AU  t=t_dyn=%.6g yr\n", \
		units.v/1.0e5, units.l/FB_CONST_AU, units.t/FB_CONST_YR);
	fprintf(stderr, "  M=%.6g M_sun  E=%.6g erg\n\n", units.m/FB_CONST_MSUN, units.E);
	
	/* move hierarchies analytically in from infinity along hyperbolic orbit */
	fb_dprintf("moving hierarchies analytically in from infinity...\n");

	m0 = hier.obj[0]->m;
	m1 = hier.obj[1]->m;
	M = m0 + m1;
	mu = m0 * m1 / M;

	vinf /= units.v;
	b /= units.l;

	double v0 = vinf / (1.0 + m0/m1);
	double v1 = vinf / (1.0 + m1/m0);
	double x_sep = 2*m0*m1*(10000) / (m0*fb_sqr(v0) + m1*fb_sqr(v1));

	Ei = 0.5 * mu * fb_sqr(vinf);

	hier.obj[0]->x[0] = - x_sep / (1.0 + m0/m1);
	hier.obj[0]->x[1] = - b / (1.0 + m0/m1); 
	hier.obj[0]->x[2] = 0.0;
	
	hier.obj[0]->v[0] = v0;
	hier.obj[0]->v[1] = 0.0; 
	hier.obj[0]->v[2] = 0.0;

	hier.obj[1]->x[0] = x_sep / (1.0 + m1/m0);
	hier.obj[1]->x[1] = b / (1.0 + m1/m0); 
	hier.obj[1]->x[2] = 0.0;
	
	hier.obj[1]->v[0] = -v1; 
	hier.obj[1]->v[1] = 0.0; 
	hier.obj[1]->v[2] = 0.0;
	
	/* and check to see that we conserved energy and angular momentum */
	fb_cross(hier.obj[0]->x, hier.obj[0]->v, l0);
	fb_cross(hier.obj[1]->x, hier.obj[1]->v, l1);
	
	for (i=0; i<3; i++) {
		L[i] = (m0 * l0[i] + m1 * l1[i]);
		r[i] = hier.obj[1]->x[i] - hier.obj[0]->x[i];
	}

	E = - m0 * m1 / fb_mod(r) + 0.5 * (m0 * fb_dot(hier.obj[0]->v, hier.obj[0]->v) + \
					m1 * fb_dot(hier.obj[1]->v, hier.obj[1]->v));

	fb_dprintf("L0=%.6g DeltaL/L0=%.6g DeltaL=%.6g\n", mu*b*vinf, fb_mod(L)/(mu*b*vinf)-1.0, fb_mod(L)-mu*b*vinf);
	fb_dprintf("E0=%.6g DeltaE/E0=%.6g DeltaE=%.6g\n\n", Ei, E/Ei-1.0, E-Ei);

	/* store the initial energy and angular momentum*/
	Ei = fb_petot(&(hier.hier[hier.hi[1]]), hier.nstar) + fb_ketot(&(hier.hier[hier.hi[1]]), hier.nstar) +
		fb_einttot(&(hier.hier[hier.hi[1]]), hier.nstar);
	fb_angmom(&(hier.hier[hier.hi[1]]), hier.nstar, Li);
	fb_angmomint(&(hier.hier[hier.hi[1]]), hier.nstar, Lint);
	for (j=0; j<3; j++) {
		Li[j] += Lint[j];
	}

	/* integrate along */
	fb_dprintf("calling fewbody()...\n");
	
	/* call fewbody! */
	retval = fewbody(input, units, &hier, &t, rng);

	/* print information to screen */
	fprintf(stderr, "OUTCOME:\n");
	if (retval.retval == 1) {
		fprintf(stderr, "  encounter complete:  t=%.6g (%.6g yr)  %s  (%s)\n\n",
			t, t * units.t/FB_CONST_YR,
			fb_sprint_hier(hier, string1),
			fb_sprint_hier_hr(hier, string2));
	} else {
		fprintf(stderr, "  encounter NOT complete:  t=%.6g (%.6g yr)  %s  (%s)\n\n",
			t, t * units.t/FB_CONST_YR,
			fb_sprint_hier(hier, string1),
			fb_sprint_hier_hr(hier, string2));
	}

	fb_dprintf("there were %ld integration steps\n", retval.count);
	fb_dprintf("fb_classify() was called %ld times\n", retval.iclassify);
	
	fprintf(stderr, "FINAL:\n");
	fprintf(stderr, "  t_final=%.6g (%.6g yr)  t_cpu=%.6g s\n", \
		t, t*units.t/FB_CONST_YR, retval.tcpu);

	fprintf(stderr, "  L0=%.6g  DeltaL/L0=%.6g  DeltaL=%.6g\n", fb_mod(Li), retval.DeltaLfrac, retval.DeltaL);
	fprintf(stderr, "  E0=%.6g  DeltaE/E0=%.6g  DeltaE=%.6g  DeltaE_GW=%.6g\n", Ei, retval.DeltaEfrac, retval.DeltaE, retval.DeltaE_GW/Ei);
	fprintf(stderr, "  Rmin=%.6g (%.6g RSUN)  Rmin_i=%d  Rmin_j=%d\n", \
		retval.Rmin, retval.Rmin*units.l/FB_CONST_RSUN, retval.Rmin_i, retval.Rmin_j);
	fprintf(stderr, "  Nosc=%d (%s)\n", retval.Nosc, (retval.Nosc>=1?"resonance":"non-resonance"));


	/* free GSL stuff */
	gsl_rng_free(rng);

	/* free our own stuff */
	fb_free_hier(hier);

	/* done! */
	return(0);
}
