
#include<complex>
#include<cmath>
extern "C"
{
    double f_b[] = {0.130,0.165,0.094,0.122,0.13,0.118,0.084,0.156};

    double d = -8.5;
    double Rsun = -8.5; //position of the sun in kpc
    double bring = 0.1, bring_unc = 0.1;	// ring at 3 kpc < rho < 5 kpc
    double hdisk = 0.4, hdisk_unc = 0.03;	// disk/halo transition
    double wdisk = 0.27, wdisk_unc = 0.08; // transition width
    double b_val[] = {0.1,3.,-0.9,-0.8,-2.,-4.2,0.,2.7};	// field strength of sM_PIral arms at 5 kpc
    
    //Jansson12b & Jansson12c

    double b_unc[] = {1.8,0.6,0.8,0.3,0.1,0.5,1.8,1.8};
    double rx[] = {5.1,6.3,7.1,8.3,9.8,11.4,12.7,15.5}; // dividing lines of sM_PIral lines
    double idisk = 11.5 * M_PI/180.;	// sM_PIral arms opening angle
    // Halo
    double Bn = 1.4, Bn_unc = 0.1 ;// northern halo

    //Jansson12c

    double Bs = -1.1, Bs_unc	= 0.1; // southern halo

    //Jansson12c

    double rhon = 9.22, rhon_unc = 0.08;// transition radius north
    double rhos = 16.7, rhos_unc = 0.;// transition radius south, lower limit
    double whalo = 0.2, whalo_unc	= 0.12;	// transition width
    double z0 = 5.3, z0_unc = 1.6;// vertical scale height
    // Out of plaxe or "X" component
    double BX0 = 4.6, BX_unc = 0.3;//field strength at origin
  
    //Jansson12b & Jansson12c  

    double ThetaX0 = 49. * M_PI/180., ThetaX0_unc	= M_PI/180.; // elev. angle at z = 0, rho > rhoXc
    double rhoXc = 4.8, rhoXc_unc = 0.2;// radius where thetaX = thetaX0
    double rhoX = 2.9, rhoX_unc = 0.1;	//exponential scale length
    // striated field
    double Bgamma = 2.92, Bgamma_unc	= 0.14; // striation and / or rel. elec. number dens. rescaling

    double L(double z,double h,double w);
    void r_log_spiral(double phi,double *result);
    void B_disk(double rho,double phi,double z,double *Bdisk);
    void B_halo(double rho,double z,double *Bhalo);
    void B_X(double rho,double z,double *BX);
    double sq(double *path, double l,double b,double b0,double b1,double b2,double b3,double b6,double* ne, double* spec_0, double* spec_1, double* spec_2, double N0, double G, double Ec, double m_a, double g_ag, double step, int size_spec, double E0, double G2);
    double P_ga(double E, double m_a, double g_ag, double* B_perp, double* psi, double* ne, double step);
    
    void B_helio(double *B_perp, double *psi, double *path,double l,double b)
    {
        double s;

        double* Bdisk = new double[3]; 
        double* Bhalo = new double[3];
        double* BX = new double[3];
        
        for(int i=0;i<100;i++)
        {
            s = path[i];
            //Galacto-centric cylindrical co-ordinates in terms of helio-centric co-ordinates
            double cb = cos(b),sb = sin(b);
            double r = sqrt(s*s*cb*cb + d*d + 2*s*d*cos(l)*cb);
            double p = atan2(s*sin(l)*cb,(s*cos(l)*cb+d));
            double z = s*sb;
            
            B_disk(r, p, z,Bdisk); 
            B_halo(r, z,Bhalo);
            B_X(r,z,BX);

            double B_r = Bdisk[0] + Bhalo[0] + BX[0]; 
            double B_p = Bdisk[1] + Bhalo[1] + BX[1];
            double B_z = Bdisk[2] + Bhalo[2] + BX[2];

            double clp = cos(l-p);
            double slp = sin(l-p);
            
            //B_s = cb*(B_r*clp+B_p*slp) + B_z*sb
            double B_b = sb*(B_r*clp+B_p*slp) - B_z*cb;
            double B_l = -B_r*slp + B_p*clp;

            B_perp[i] = sqrt(B_b*B_b + B_l*B_l);
            psi[i] = atan2(B_b,B_l);
        }
        delete[] Bdisk;
        delete[] Bhalo;
        delete[] BX;
    }

    double sq(double *path, double l,double b,double b0,double b1,double b2,double b3,double b6,double* ne, double* spec_0, double* spec_1, double* spec_2, double N0, double G, double Ec, double m_a, double g_ag, double step, int size_spec, double E0, double G2)
    {   
        /** --- Planck Update BEGIN --- **/
        
        /** --- Jansson12b --- **/

        b_val[5] = -3.5; 
        BX0 = 1.8; 

        /** --- Jansson12c --- **/

        b_val[1] = 2; 
        b_val[3] = 2; 
        b_val[4] = -3;
        Bs = -0.8;
        Bn = 1; 
        BX0 = 3;
        
        b_val[7] = 0;
        for(int i=0;i<7;i++)
        {
            b_val[7] += -f_b[i]*b_val[i]/f_b[7]; // Conservation B-field flux --- See 5.1.1 of Jansson&Fahrar 2012 - 1204.3662
        }

        /** --- Planck Update END --- **/

        b_val[0] = b0;
        b_val[1] = b1;
        b_val[2] = b2;
        b_val[3] = b3;
        b_val[6] = b6;

        // UPDATE b7

        b_val[7] = 0;
        for(int i=0;i<7;i++)
        {
            b_val[7] += -f_b[i]*b_val[i]/f_b[7]; // Conservation B-field flux --- See 5.1.1 of Jansson&Fahrar 2012 - 1204.3662
        }

        double *B_perp = new double[100];
        double *psi = new double[100];

        B_helio(B_perp,psi,path,l,b);

        double E,dnde=0.,s2 = 0.,sigma;
        int i;

        for(i=0;i<size_spec;i++)
        {
            E = spec_0[i]/1000.;
            sigma = sqrt(spec_2[i]*spec_2[i] + pow(0.024*spec_1[i],2));

            if(m_a > 0. ) 
            {
                dnde = N0 * pow(10,-9) *pow((E/E0),-G) * exp(-pow((E/Ec),G2))*P_ga(E,m_a,g_ag,B_perp,psi,ne,step)*E*E *pow(10,6)/624150.934 ;
            }
            else 
            {
                dnde = N0 * pow(10,-9) *pow((E/E0),-G) * exp(-pow((E/Ec),G2))*E*E *pow(10,6)/624150.934;                
            }

            s2 = s2 + pow((dnde-spec_1[i])/sigma,2) ;
        }
        delete[] B_perp;
        delete[] psi;
        
        return s2;
    }

    double L(double z,double h,double w)
    {
        return (1./(1. + exp(-2. * (fabs(z) - h) / w)));
    }

    void B_disk(double rho,double phi,double z,double *Bdisk)
    {
        Bdisk[0] = 0.;
        Bdisk[1] = 0.;
        Bdisk[2] = 0.;

        int i,narm;
        double min;
        if((rho >= 3.) && (rho < 5.)) 
        {
            Bdisk[1] = bring;
        }
        if((rho >= 5.) && (rho <= 20.)) 
        {
            double rls[32]; 
            for(i=0;i<8;i++)
            {
                rls[i] = rx[i]*exp((phi - 3.*M_PI) / tan(M_PI/2. - idisk));
                rls[i+8] = rx[i]*exp((phi - M_PI) / tan(M_PI/2. - idisk));
                rls[i+16] = rx[i]*exp((phi + M_PI) / tan(M_PI/2. - idisk));
                rls[i+24] = rx[i]*exp((phi + 3.*M_PI) / tan(M_PI/2. - idisk));
            }
            for(i=0;i<32;i++)
            {
                rls[i] = rls[i] - rho;
                if(rls[i] < 0.) rls[i] = 1e10;
                if(rls[i]<min || i==0) 
                {
                    min = rls[i];
                    narm = i;
                }
            }
            narm = narm%8;
            Bdisk[0] = sin(idisk)*b_val[narm] * (5. / rho);
            Bdisk[1] = cos(idisk)*b_val[narm] * (5. / rho);
        }

        for(i=0;i<2;i++)
            Bdisk[i] *= (1 - L(z,hdisk,wdisk));
    }

    void B_halo(double rho,double z,double *Bhalo)
    {
        Bhalo[0] = 0.;
        Bhalo[1] = 0.;
        Bhalo[2] = 0.;
        
        int zp=0,zm=0;
        if ( z != 0. )
        {
            if(z>0.)
                zp = 1;
            else
                zm = 1;
            Bhalo[1] = exp(-fabs(z)/z0) * L(z,hdisk,wdisk) * 
                        ( Bn * (1 - L(rho, rhon, whalo)) * zp 
                        + Bs * (1 - L(rho, rhos, whalo)) * zm );
        }
    }

    void B_X(double rho,double z,double *BX)
    {
        BX[0] = 0.;
        BX[1] = 0.;
        BX[2] = 0.;
        
        int zp=0,zm=0;
        double theta, rho_p0,bx;
        if(sqrt(rho*rho + z*z) >= 1.)
        {
            if(z>=0.) zp = 1;
            else zm = 1;
            double rho_p = rho * rhoXc/(rhoXc + fabs(z) / tan(ThetaX0));

            if(rho_p > rhoXc) 
            {
                rho_p0	= rho  - fabs(z) / tan(ThetaX0);
                bx = (BX0 * exp(-rho_p0 / rhoX)) * rho_p0/rho;
                theta	= ThetaX0;
            }
            if(rho_p <= rhoXc) 
            {
                bx	= (BX0 * exp(-rho_p / rhoX)) * (rho_p/rho)* (rho_p/rho);
                theta	= atan2(fabs(z),(rho - rho_p));
            }
            if(z==0.) theta = M_PI/2.;
            
            BX[0] = bx * (cos(theta) * (zp) + cos(M_PI - theta) * (zm));
            BX[2] = bx * (sin(theta) * (zp) + sin(M_PI - theta) * (zm));
            
        }
    }

    double P_ga(double E, double m_a, double g_ag, double* B_perp, double* psi, double* ne, double step)
    {   
        double cp, sp, cp2, sp2, scp, D_ag, D_aa, D_pl, L1, L2, L3, alpha, ca, sa, ca2, sa2, sca, sa2sp2, sa2spcp, sacasp, sa2cp2,sacacp,ca2sp2, ca2spcp, ca2cp2;
        int i1,i2;
        std::complex<double> el1,el2,el3;
        int i;
        double result;

        std::complex<double> U[3][3]={{1.,0,0,},{0,1.,0},{0,0,1.}};
        std::complex<double> Uc[3][3]={{1.,0,0,},{0,1.,0},{0,0,1.}};
        std::complex<double> AB[3][3];
        std::complex<double> e1cp2, e1scp, e1sp2,e2sa2sp2,e2sa2spcp,e2sacasp,e2sa2cp2,e2sacacp,e2ca2,e3ca2sp2,e3ca2spcp,e3sacasp,e3ca2cp2,e3sacacp,e3sa2;
        for(i=0;i<100;i++)
        {
            cp = cos(psi[i]);
            sp = sin(psi[i]);
            cp2 = cp*cp;
            sp2 = sp*sp;
            scp = sp*cp;
        
            D_ag = 1.52*pow(10,-2)*(g_ag*B_perp[i]);
            D_aa = -7.8*pow(10,-2)*(m_a*m_a/E) ;
            D_pl = -1.1*pow(10,-7)*(ne[i]/E)*1000;
            
            L1 = D_pl*step;
            L2 = 1./2.*(D_pl+D_aa-sqrt(pow(D_pl-D_aa,2) + 4*D_ag*D_ag))*step;
            L3 = 1./2.*(D_pl+D_aa+sqrt(pow(D_pl-D_aa,2) + 4*D_ag*D_ag))*step;
            
            alpha = atan2(2*D_ag,(D_pl-D_aa))/2.;
            
            ca = cos(alpha);
            sa = sin(alpha);
            
            el1 = {cos(L1),sin(L1)};
            el2 = {cos(L2),sin(L2)};
            el3 = {cos(L3),sin(L3)};
        
            ca2 = ca*ca;
            sa2 = sa*sa;
            sca = sa*ca;
        
            sa2sp2 = sa2*sp2;
            sa2spcp = sa2*scp;
            sacasp = sca*sp;
            sa2cp2 = sa2*cp2;
            sacacp = sca*cp;
            ca2sp2 = ca2*sp2;
            ca2spcp = ca2*scp;
            ca2cp2 = ca2*cp2;
        
            e1cp2 = cp2*el1;
            e1scp = scp*el1;
            e1sp2 = sp2*el1;
        
            e2sa2sp2 = el2*sa2sp2;
            e2sa2spcp = el2*sa2spcp;
            e2sacasp = el2*sacasp;
            e2sa2cp2 = el2*sa2cp2;
            e2sacacp = el2*sacacp;
            e2ca2 = el2*ca2;
        
            e3ca2sp2 = el3*ca2sp2;
            e3ca2spcp = el3*ca2spcp;
            e3sacasp = el3*sacasp;
            e3ca2cp2 = el3*ca2cp2;
            e3sacacp = el3*sacacp;
            e3sa2 = el3*sa2;
        
            std::complex<double> T[3][3] = {{e1cp2 + e2sa2sp2 + e3ca2sp2,e2sa2spcp-e1scp+e3ca2spcp,e3sacasp-e2sacasp},
                                            {e3ca2spcp+e2sa2spcp-e1scp,e2sa2cp2+e1sp2+e3ca2cp2,e3sacacp-e2sacacp},
                                            {e3sacasp-e2sacasp,e3sacacp-e2sacacp,e2ca2+e3sa2}};
                                        
            for(i1=0;i1<3;i1++)
            {
                for(i2=0;i2<3;i2++)
                {
                    AB[i1][i2] = U[i1][0]*T[0][i2] + U[i1][1]*T[1][i2] + U[i1][2]*T[2][i2]; 
                }
            }
            
            for(i1=0;i1<3;i1++)
            {
                for(i2=0;i2<3;i2++)
                {
                    U[i1][i2] = AB[i1][i2];
                    Uc[i2][i1] = conj(U[i1][i2]);
                }
            }
        }
        double Rf[3][3] = {{1,0,0},{0,1,0},{0,0,0}};
        double Ri[3][3] = {{0.5,0,0},{0,0.5,0},{0,0,0}};
        
        
        result = real((Rf[0][0]*U[0][0]+Rf[0][1]*U[1][0]+Rf[0][2]*U[2][0])*(Ri[0][0]*Uc[0][0]+Ri[0][1]*Uc[1][0]+Ri[0][2]*Uc[2][0])+ 
                    (Rf[1][0]*U[0][0]+Rf[1][1]*U[1][0]+Rf[1][2]*U[2][0])*(Ri[0][0]*Uc[0][1]+Ri[0][1]*Uc[1][1]+Ri[0][2]*Uc[2][1])+ 
                    (Rf[2][0]*U[0][0]+Rf[2][1]*U[1][0]+Rf[2][2]*U[2][0])*(Ri[0][0]*Uc[0][2]+Ri[0][1]*Uc[1][2]+Ri[0][2]*Uc[2][2])+
                    (Rf[0][0]*U[0][1]+Rf[0][1]*U[1][1]+Rf[0][2]*U[2][1])*(Ri[1][0]*Uc[0][0]+Ri[1][1]*Uc[1][0]+Ri[1][2]*Uc[2][0])+ 
                    (Rf[1][0]*U[0][1]+Rf[1][1]*U[1][1]+Rf[1][2]*U[2][1])*(Ri[1][0]*Uc[0][1]+Ri[1][1]*Uc[1][1]+Ri[1][2]*Uc[2][1])+ 
                    (Rf[2][0]*U[0][1]+Rf[2][1]*U[1][1]+Rf[2][2]*U[2][1])*(Ri[1][0]*Uc[0][2]+Ri[1][1]*Uc[1][2]+Ri[1][2]*Uc[2][2])+
                    (Rf[0][0]*U[0][2]+Rf[0][1]*U[1][2]+Rf[0][2]*U[2][2])*(Ri[2][0]*Uc[0][0]+Ri[2][1]*Uc[1][0]+Ri[2][2]*Uc[2][0])+ 
                    (Rf[1][0]*U[0][2]+Rf[1][1]*U[1][2]+Rf[1][2]*U[2][2])*(Ri[2][0]*Uc[0][1]+Ri[2][1]*Uc[1][1]+Ri[2][2]*Uc[2][1])+ 
                    (Rf[2][0]*U[0][2]+Rf[2][1]*U[1][2]+Rf[2][2]*U[2][2])*(Ri[2][0]*Uc[0][2]+Ri[2][1]*Uc[1][2]+Ri[2][2]*Uc[2][2]));
            
        return result;

    }

}