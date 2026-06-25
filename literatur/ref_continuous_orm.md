# REFERANS: Noninvasively improving the orbit-response matrix while continuously correcting the orbit (arXiv:2104.05300)

> pdftotext ile çıkarıldı (özgünlük karşılaştırması için).

---

                                                       Noninvasively improving the orbit-response matrix while
                                                                        continuously correcting the orbit

                                                                                          Ingvar Ziemann
                                                                        KTH, Royal Institute of Technology, Stockholm


                                                                                          Volker Ziemann
arXiv:2104.05300v2 [physics.acc-ph] 1 Jul 2021




                                                                                         Uppsala University
                                                                                        (Dated: July 2, 2021)

                                                       Based on continuously recorded beam positions and corrector excitations from, for
                                                       example, a closed-orbit feedback system we describe an algorithm that continuously
                                                       updates an estimate of the orbit response matrix. The speed of convergence can
                                                       be increased by adding very small perturbations, so-called dither, to the corrector
                                                       excitations. Estimates for the rate of convergence and the asymptotically achievable
                                                       accuracies are provided.



                                                                                   I.     INTRODUCTION


                                                   The orbit-response matrix relates changes of the dipole corrector magnets to orbit changes
                                                 that are observed on the beam position monitor system. It is of paramount importance for
                                                 maintaining stable beam positions in storage rings, which is typically accomplished by “slow”
                                                 orbit correction systems [1–4] and “fast” feedback systems [5, 6]. They either use a response
                                                 matrix generated from a computer model of the accelerator or a measured matrix found by
                                                 varying one corrector at a time and observing the ensuing changes with the beam position
                                                 monitor (BPM) system.
                                                   As a matter of fact, comparing the measured matrix with a matrix derived from a com-
                                                 puter model, as discussed in [7–9], makes it possible to track down deficient hardware, such
                                                 as incorrectly calibrated power supplies or scale errors on position monitors. Usually, the
                                                 response matrix is measured in dedicated shifts, labeled “machine development,” where the
                                                 excitation of one corrector after the other is varied and the resulting changes of the posi-
                                                 tions on the orbit monitor system are recorded, which is commonly referred to as “open
                                                 loop” measurements. In this report, we discuss an algorithm that complements the exist-
                                                                                          2

ing methods. It requires no dedicated beam time and slowly improves an estimate of the
response matrix quasi for free by using information from the “closed loop” orbit feedback
system. The procedure, based on a recursive least-squares algorithm [10, 11], is completely
non-invasive and can run while operating the accelerator in production mode—producing
luminosity in a collider, or photons in a light source. It has the remarkable property that
the error bars asymptotically approach zero as the estimated response matrix approaches
the “real” response matrix. The algorithm is, however, slow, because it “learns from noise”
but might nevertheless prove useful to continuously improve the response matrix at times
normally not accessible for machine improvement. This opens the possibility to track down
very slow changes of hardware parameters when post-processing the response matrix with,
for example, LOCO [8].
   This report is organized as follows: in the next section we develop the algorithm, fol-
lowed by Section III, where we introduce a simple model storage ring used to illustrate it.
In Section IV we introduce dithering as a way to speed up the algorithm, before we explore
its convergence properties, both during the early stages in Section V, and in the asymp-
totic regime in Section VI. Before concluding, we address a number of technical issues and
extensions to the algorithm in Section VII.


                                II.   THE ALGORITHM


   The response matrix B with matrix elements B ij relates the change in excitation uj of
steering magnet j with 1 ≤ j ≤ m to a change of the beam position xi with 1 ≤ i ≤ n
on monitor i. Here superscripts denote different monitors and correctors. We will use the
notation from quantum mechanics with bra states denoting column vectors and ket states
denoting row vectors, which will prove convenient later on. We thus collectively denote the
values of all n BPM by |xi and the m correctors by |ui. Correcting the orbit then means
to to add a perturbation B |ui to the orbit |xi that minimizes the residual orbit |x̃i after
correction. It is given by
                                 |x̃i = |xi + B |ui + |wi ,                              (1)

where |wi describes noise in the system, for example, due to ground motion or BPM noise.
When correcting the orbit, we have to find corrector excitations |ui that minimize hx̃|x̃i.
One problem is that we do not have complete knowledge of the system matrix B. All we
                                                                                                3

do know is a more or less accurate estimate B̃ that was previously derived from a computer
model or from measurements and use that when correcting the orbit.
   Assuming that the position monitors report values |xi and furthermore assuming that the
desired orbit is centered around zero, allows us to calculate the desired corrector excitations
|ui from inverting B̃, the approximation of B from Equation 1. If B̃ is square (n = m)
and invertible this is just the matrix-inverse −B̃ −1 , where the minus sign ensures that
the effect of the correctors cancels the observed orbit. If B̃ is over-determined (n > m)
this is accomplished by the Moore-Penrose pseudo inverse −(B̃ > B̃)−1 B̃ > , which follows
                           >              
from minimizing |xi + B̃ |ui     |xi + B̃ |ui with respect to |ui. If B̃ is under-determined
(n < m) it can be inverted using singular value decomposition. In general, we denote the
linear dependence of the corrector excitations on the observed orbit |xi by the “correction”
matrix K, such that |ui = −K |xi.
   Our task is now to extract information from repeatedly correcting the orbit and correlat-
ing the orbit change with the used corrector changes |ui. To this end we note that the noise
|wi and the mismatch of the “real” accelerator model B and B̃ from which K is derived,
causes the correction to be imperfect. We model this dependence by the dynamical system

                 |xt+1 i = |xt i + B |ut i + |wt i   with     |ut i = −K |xt i ,              (2)

where the subscript t denotes a discrete time step from one iteration of the orbit correction
to the next. We assume that the noise |wt i is Gaussian and characterized by the expectation
value E{|ws i hwt |} = σw2 Cδst . Here σw2 C is the spatial covariance matrix, where σw is the rms
magnitude and C describes correlations among different BPM. In Appendix D we will return
to the general case, but assume C to be a n × n unit matrix in the main text. Furthermore,
δst is the Kronecker delta, which implies that we treat noise to be uncorrelated from one
iteration to the next. Note also that the effect of power supply noise |ṽt i added to |ut i is
equivalent to additional noise on the monitors with magnitude B |ṽt i . In Equation 2 we
implicitly omit fast time-dependent effects, such as latency in the power supplies or the
computation chain as well as the effect of eddy currents. In Section VII we briefly discuss
how to include these effects, but in the main text all transient effects are assumed to have
settled to a new equilibrium from one iteration to the next. Now the interpretation of
Equation 2 is straightforward: the system responds with the “real” response matrix B to a
change of the corrector excitation by |ut i that was calculated with the approximative inverse
                                                                                                 4

K and the orbit |xt i. At the same time, noise enters the system through |wt i, such that
the residual orbit |xt+1 i after the correction is not necessarily equal to zero. Iterating the
orbit correction, which is what orbit feedback systems essentially do, can now be modeled
by iterating the system described by Equation 2.
   In order to find an estimate B̂ of the system matrix B one row—corresponding to a
particular BPM i—at a time, we construct linear systems of equations for each time step and
solve the resulting sequence of equations with a recursive least-squares algorithm [10, 11]. To
set up the equations, for the time being, we ignore the noise |wt i and formulate Equation 2
for this BPM as a constraint for B̂. Writing the constraints over consecutive readings xis ,
we find                                                                                  
                                                     u1 ,                          B̂  i1
                                                    s 
                                                     ..                            .. 
                                                                                     
             xis+1 − xis = B̂ i1 , . . . , B̂ im     .,  =       u1s . . . um           ,    (3)
                                                                                 
                                                                              s      .
                                                                                      
                                                        m                             im
                                                      us                           B̂
where the second equality follows from exchanging the order of writing the scalar product
of row i of B̂ and corrector excitations ujs . In the next step we assemble multiple copies of
this equation from different times 1 ≤ s ≤ T in the form of a matrix
                                                                 
               xi2 − xi1        u11 . . . um
                                           1   B̂  i1
                                                                 B̂  i1

                   ..                 ..       ..            .. 
                                                                   
                                                                               i:
                            =                             = U      .  = UT |B̂ i               (4)
                          
                   .                .         .        T 
                                                                 
             xiT +1 − xiT       u1T . . . um
                                           T     B̂ im           B̂ im

and denote the matrix containing the corrector excitations ujs by UT , which thus contains the
excitations of all correctors stacked one by one on top of the other. Likewise, the vector on
the left-hand side contains the orbit differences that each of the steering magnet excitations
causes. If we now record BPM positions and corresponding corrector excitations for a long
time T , the system of equations in Equation 4 is vastly over-determined, provided that the
noise really affects all possible degrees of freedom of the system, which implies that in general
the covariance matrix C must have full rank. Since we assume C to be the unit matrix, this
is the case and we can solve Equation 4 in the least-squares sense with the pseudo-inverse
mentioned above. We find
                                                                                
                                  B̂ i1                                xi2 − xi1
                                  T 
                                     ..          −1 >       ..
                                                                     
                                             >
                     |B̂Ti: i =      .  = UT UT    UT              ,                        (5)
                                                                   
                                                                .
                                                                  
                                      im                   i       i
                                   B̂T                    xT +1 − xT
                                                                                                    5


which provides an estimate for row i of the matrix B̂Ti: after T iterations of the orbit correc-
tions. Repeating this procedure for all BPMs provides us with an estimate for the complete
system matrix B̂T .
   In passing, we point out that Equation 5 describes a linear map from the vector with the
position differences on the right-hand side onto the vector with row i of B̂T , which allows
us to calculate an empirical (data-driven) covariance matrix of the B̂T from the covariance
matrix of the position difference, which is 2σw2 times the unit matrix. The error bars σ(B̂) of
the fitted B̂T are therefore approximately given by the square root of the diagonal elements
               −1
of 2σw2 UT> UT     , which will prove useful later on.
   Calculating the pseudo-inverse of UT for more and more iterations becomes numerically
very expensive. There is, however, an elegant way of iteratively updating the pseudo-inverse
                                                                                 −1
using the Sherman-Morrison [12] formula. It is based on updating PT = UT> UT         and B̂T
as the matrix UT grows one row at a time by adding the row vector huT +1 | = (u1T +1 , . . . , um
                                                                                                T +1 )

to it. This entails that we can write PT−1     −1
                                         +1 = PT + |uT +1 i huT +1 |. In Appendix A we show

that its inverse is given by

                                                    PT |uT +1 i huT +1 | PT
                                  PT +1 = PT −                              .                     (6)
                                                    1 + huT +1 |PT |uT +1 i
                            −1
With PT +1 = UT>+1 UT +1          known, we can calculate an updated approximation [13] of the
response matrix B̂T +1 from
                                                                           
                                          |xT +2 i − |xT +1 i − B̂T |uT +1 i huT +1 | PT
                     B̂T +1 = B̂T +                                                        .      (7)
                                                    1 + huT +1 |PT |uT +1 i

We refer to Appendix B and [13] for the derivation. Note that in Equations 6 and 7 the
right-hand sides only depend on PT and B̂T from the previous iteration, the new corrector
excitations |uT +1 i, and the most recent change in the orbit |xT +2 i − |xT +1 i. These two
equations now allow us to continuously update the response matrix while correcting the
orbit with K. All we do here is correlating the change of the orbit |xT +2 i − |xT +1 i with the
corrector excitations |uT +1 i that cause this change and then update our approximation of
B̂ in the process.
   With the basic algorithm worked out, we simulate its performance in the following section.
                                                                                                 6




FIG. 1: Left: the beta functions of one cell. Right: the beta functions of the ring where the focal
lengths of all quadrupoles are randomly perturbed by 5 % which causes a moderate beating.


                                     III.   SIMULATION


   In order to test the algorithm we prepared response matrices for a small ring consisting
of ten FODO cells each having phase advances of µx /2π = 0.228 and µy /2π = 0.238 in the
horizontal and vertical plane respectively. The tunes of the ring therefore are Qx = 2.28 and
Qy = 2.38. Moreover, there are two 18-degree sector dipole magnets in each cell. The beta
functions of one cell are shown on the left-hand side in Figure 1. We place a corrector and a
BPM at the same location as the (thin-lens) focusing quadrupole, which then accounts for
ten correctors and ten BPM, each. In order to keep the simulation transparent, we calculate
the response matrix B between these correctors and BPM in the horizontal plane only. Most
of the simulations are done for equal numbers of correctors and monitors; we address other
cases in Section VII. The response matrix derived from this unperturbed ring is the “ideal”
                                                                        −1
                                                                     >
response matrix B̃ that we use to derive the correction matrix K = B̃ B̃     B̃ > to correct
the orbit. In order to simplify the theoretical analysis in Section VI, in the remainder of
this report we confine ourself to a constant correction matrix B̃. If, instead we were to
use the constantly updating B̂T for the correction, the algorithm would be adaptive. In
order to determine the “real” response matrix B, we randomly vary the focal lengths of the
quadrupoles with a rms of 5 % and re-calculate the response matrix B for the perturbed
ring. We take notice that the rms magnitude of the response coefficients is 6.6 m/rad. In
order to quantify the estimation error bT = B̂T − B after T iterations, we introduce the
                                                                                                   7




                        p
FIG. 2: The rms orbit       hx|xi (top) and the rms of all matrix elements of B − B̂ (bottom) for 105
iterations. The noise level was chosen σw = 0.1 mm. The upper plot reproduces the noise level and
the lower graph shows that the response-matrix estimate B̂ slowly converges towards B.


rms value of bT , calculated over all matrix elements as discrepancy |bT |rms . It can also be
calculated from                          v
                                         u                         
                                         u Trace (B̂ − B)> (B̂ − B)
                                         t          T         T
                            |bT |rms =                                    .                      (8)
                                                       nm
Evaluating the initial value |b0 |rms for our model storage ring, we find that it is approximately
0.3 m/rad which accounts for a 5 % rms deviation of the response matrix coefficients. The
simulations are based on Matlab scripts that use beam optics functions from [14]. The code
illustrating one iteration of the algorithm is reproduced and commented in Appendix C.
  Running the simulation for 105 iterations, which takes a few seconds on a desktop com-
                                                                          p
puter, produces Figure 2, which shows the evolution of the rms orbit σx2 = hx|xi and the
discrepancy |bT |rms between the “real” and the estimated response matrix. We initialized
the estimate for B̂0 with the response matrix for the ring without quadrupole gradient errors
and P0 with the unit matrix. The rms amplitude of the noise σw was chosen to be 0.1 mm.
                                                                                              8




FIG. 3: The rms value of all matrix elements of B − B̂ after 105 iterations as a function of the
noise level σw .


Note that the upper plot, which shows the rms orbit σx for the duration of the simulation
clearly verifies this; the mean is close to 0.1 mm. At the same time, the discrepancy |bT |rms ,
shown on the lower plot, is approximately halved to a final value |bf |rms = 0.168 m/rad,
which shows that the algorithm works.
   Repeating the same simulation (always for 105 iterations) for different values of σw and
recording the final discrepancy |bf |rms produces the plot shown in Figure 3. Here we find that
increasing noise levels are beneficial for the rate of convergence, up to about σw = 0.5 mm,
where the induced changes in b during one iteration become comparable to the magnitude
of b. We need to stress that the plotted values are those reached after 105 iterations. They
are not the asymptotic levels.
   The algorithm is rather stable. We ran simulations where we initialized B̂ with random
matrices or other made-up starting guesses. The algorithm, after an initial transient period,
always converged towards the “real” matrix B.
   We point out that the convergence depends on the noise level, where more noise moves
the correctors around more and actually improves the convergence, but the rate is still rather
slow, on the order of several 105 iterations, which would correspond to about three hours
real time, provided that the feedback operates at an update rate of 10 per second. Moreover,
                                                                                                 9




FIG. 4: Simulations based on parameters used for Figure 2. Left: round-robin dithering with
20 µrad active from 20000 to 40000 iterations, which temporarily increases the rms orbit σx but
significantly helps to reduce the discrepancy |brms |. Right: dithering with the same parameters is
active all the time, which increases σx all the time, but reduces |brms | even further.


the asymptotically achievable discrepancy is of considerable interest. We will address these
topics below after having introduced the effect of additional corrector perturbations.


                                       IV.    DITHERING


   Varying the corrector excitations one at a time, either systematically or sinusoidally [15–
17], in order to determine the response matrix is used in practically all accelerators. More-
over, continuously varying correctors very little such that the detrimental effect on the orbit
is negligible, so-called dithering, was successfully used [18–20] to optimize the performance
of a number of accelerators. We implement dithering in our simulation by adding a per-
turbing vector |zt i to |ut i when correcting the orbit in Equation 2, which therefore becomes
|ut i = −K |xt i + |zt i. The rest of the simulation remains unaffected; any changes of |zt i
and consequently of |ut i are consistently accounted for in the updates of PT and B̂T in
Equations 6 and 7.
   In the simulations, shown in Figure 4, we chose to add 20 µrad to the excitation z k to one
corrector k at a time in a round-robin fashion and record the rms orbit and the discrepancy
for 105 iterations. The plot on the left-hand side shows the simulation where the dithering
                                                                                                   10




FIG. 5: Left: the rms orbit σx (dashed black) and the discrepancy |brms | (solid red) as a function of
the dither amplitude, which allows us to assess the trade-off between spoiling the orbit and learning
the response matrix. Right: the solid red line shows |bf |rms plotted versus σx . The black dashed
line shows the effect of purely random variations, already shown in Figure 3, for comparison.


was turned on between 20000 and 40000 iterations. We clearly see that the rms orbit
increases from 0.10 to 0.16 mm during this period, which is consistent with expectations,
because the rms value of the B of 6.6 m/rad and 20 µrad additional excitation results in an
additional rms orbit variation of 0.13 mm, which, added in quadrature to σw = 0.1 mm, gives
about 0.16 mm. We also observe on the lower plot that the discrepancy |bT |rms is significantly
reduced and conclude that temporarily adding dithering helps to improve our knowledge of
the response matrix. Note that no additional processing of the data is necessary. The
algorithm learns whenever it gets the chance to observe some variation, never mind the
source of the perturbation. Remarkably, a slammed door might be beneficial for something.
In the simulation shown on the right-hand side in Figure 4, we keep the 20 µrad round-
robin dithering on permanently and observe that the rms orbit is 0.16 mm throughout the
simulation, while the discrepancy |bT |rms is reduced sevenfold. Again, no special processing
is required.
   The left-hand plot in Figure 5 illustrates the effect of dither amplitude, shown on the
horizontal axis, on the rms orbit (dashed black) and on the discrepancy (solid red). We
clearly observe that the increasing dither amplitude increases the rms orbit σx , but at the
same time, helps to reduce the discrepancy |bf |rms . Closer inspection shows that a dither
                                                                                             11




FIG. 6: The upper panel shows the evolution of |PT |2 (solid black) and |P̂T |2 (dashed red) for
a configuration with σw = 0.1 mm and z = 20 µrad. The lower panel shows the corresponding
evolution of bT (solid black) and b̂T (dashed red).


amplitude of 16 µrad contributes to σx with the same magnitude as normal noise level σw .
This causes σx to increase by 40 %. At the same time, |bf |rms is reduced by 1/3 from
0.168 m/rad to 0.056 m/rad. This configuration is indicated by the vertical dotted line in
Figure 5.
   The right-hand plot in Figure 5 shows the data from the left-hand plot, but now plotting
the discrepancy |bf |rms versus the the rms orbit σx (solid red) and compares it to the data
from Figure 3 (dashed black). Unsurprisingly, increasing σx by dithering reduces |bf |rms
more efficiently than just increasing the natural noise level σw .


                                    V.    CONVERGENCE


   A matter of practical interest are the time scales, given by the number of iterations, before
we observe some improvement of the response matrix. We point out that the results devel-
                                                                                                12

oped in the following sections apply to all systems described by Equation 2, which includes
rings with transverse coupling and correction matrices K that use elaborate regularization
schemes. The simulations, which are based on correctors and monitors in a single transverse
plane, are only used to illustrate the general results. Let us start by analyzing the initial
behavior of the discrepancy and approximate Equation 6 by replacing |uT +1 i huT +1 | by its
expectation value E{|uT +1 i huT +1 |}, which asymptotically becomes independent of T . We
therefore use E{|uT i huT |} instead, which depends on |xT i via |uT i = −K |xT i and calculate

                      |xT i = (1 − BK) |xT −1 i + |wT −1 i
                                                 T −1
                                                 X
                                      T
                            = (1 − BK) |x0 i +        (1 − BK)s |wT −s−1 i ,                    (9)
                                                         s=0

where the second equality results from iterating the first equality. Since the spectral radius
ρ(Λ) with Λ = 1 − BK is much less than unity, the influence of the initial |x0 i “dies out”
for large T and we can omit the first term from the sum. Inserting |xT i in |uT i = −K |xT i,
we obtain
                             (    "T −1                # "T −1                  #    )
                                                                              r
                                   X                      X
                                          Λs |wT −s−1 i        hwT −r−1 | Λ>      K > + o(1)
                                                                             
       E{|uT i huT |} = E K
                                    s=0                        r=0
                             T −1 X
                                  T −1
                             X                                             r
                       = K               Λs E {|wT −s−1 i hwT −r−1 |} Λ>        K > + o(1)     (10)
                             s=0 r=0
                               T −1
                               X               s
                       = σw2 K           ΛΛ>        K > + o(1) ,
                                 s=0

where we used that the expectation value of the Gaussian noise is E{|ws i hwt |} = σw2 δst 1.
Moreover, o(1) denotes a quantity that vanishes in the limit of large T . The smallness of
ρ(Λ) implies that only the term with s = 0 in the sum in Equation 10 contributes and we
have E {|uT i huT |} ≈ σw2 KK > , which is indeed independent of T . We include round-robin
                                                                                    2
dithering with amplitude z through the m correctors by adding a term zm 1, because dithering
is uncorrelated to the noise and after m iterations dithering contributes a unit matrix. We
thus just “spread out” this unit matrix to the individual iterations when diving by m. We
therefore introduce
                                                                 2
                                                         z
                                          Q = σw2 KK > +             1                         (11)
                                                               m
                                                                                               13




FIG. 7: Time scales of the convergence, determined from the inverse eigenvalues of Q for dithering
amplitudes z from 0 to 30 mrad. In all cases we use σw = 0.1 mm. The dashed red line shows the
                                2σ          > + z 2 /m corresponding to the smallest eigenvalue
                                             
time scale 1/λmin with λmin = σw   min KK

of Q.


to represent the average effect of the orbit correction and dithering when updating the
“averaged” P̂T in Equation 6, which then reads

                                                     P̂T QP̂T
                               P̂T +1 = P̂T −                  .                           (12)
                                                1 + Trace QP̂T

Note that Equation 12 is a deterministic equation that describes the averaged updating
of P̂T . In the simulation we update P̂T in parallel to its “stochastic brethren” PT and
find that they are extremely close, both with and without dithering. The upper panel in
Figure 6 shows an example with σw = 0.1 mm and z = 20 µrad, which corresponds to the
configuration also displayed on the right-hand side in Figure 4. The solid black curve is
produced by a numerical simulation with simulated random noise and the dashed red curve
shows the result of the deterministic simulation, based on Equations 11 and 12.
   We point out that Q is the only parameter in the dynamics described by Equation 12. In
order to simplify the analysis somewhat, we neglect the trace in the denominator, which is
                                                                      
practically always much smaller than 1, which results in 1 + Trace QP̂T ≈ 1 and allows us
to write the equation as P̂T +1 = P̂T − P̂T QP̂T . Moreover, Q is symmetric by construction and
                                                                                               14

we can choose a coordinate system in which Q is diagonal with eigenvalues λj = σw2 σj +z 2 /m,
where σj are the eigenvalues of KK > , such that Q = ODO> with D = diag(λ1 , . . . , λm )
and an orthogonal matrix O. Also the starting guess for P0 is the unit matrix and is
diagonal, such that Equation 12 can be written as m independent equations for each of the
m diagonal elements xj,T of P̂T . Each eigenvalue thus corresponds to one mode that describes
the dynamics of the convergence process. In the following, we consider one mode at a time
and omit a second index j = 1 . . . , m from x and λ to make the equations easier to read.
We therefore obtain xT +1 = xT − λx2T or its continuous approximation dxT /dT = −λx2T for
each mode. This equation has the solution

                                                  x0
                                        xT =             .                                   (13)
                                               1 + x0 λT

Numerically x0 has the value of unity, because P̂0 is the unit matrix, but we leave it in place to
keep track of the units of x0 which are 1/mrad2 . We thus find that the inverse eigenvalues
1/λ of the matrix Q determine the time scales of the convergence of the process. Note,
however, that the time dependence is inversely proportional to T , rather than exponential,
and is therefore slow.
   Figure 7 shows the time scales 1/x0 λj with j = 1, . . . , m for dither amplitudes z between
0 and 30 mrad, while σw is always 0.1 mm. The dashed red line shows the time scale of the
slowest mode and is given by the smallest eigenvalue λmin = σw2 σmin KK > + z 2 /m. Here
                                                                        

σmin [·] is the smallest eigenvalue of the matrix in the argument. The rms orbit variation
σx approximately doubles in this range. We observe that there is always one very small
eigenvalue, which leads to a very long time scale. Dithering mostly helps to reduce this
long time scale from 1.6 × 105 to about 20000 iterations. At this point we remind ourselves,
                                                           −1
following the discussion from Section 2, that PT = UT> UT       determines the error bars of
B̂T . Since Equation 13 implies PT ∝ 1/T for T ' 1/x0 λ we find that the error bars of B̂T
          √
have a 1/ T dependence.
   It remains to analyze the time scales of the convergence of the values of B̂T to B, which
is described by Equation 7. We note that |xT +2 i − |xT +1 i = B |uT +1 i was caused by the
corrector values |uT +1 i, such that we arrive at
                                              |u i hu | P
                                                      T +1     T +1   T
                   B̂T +1 − B = B̂T − B − B̂T − B                          ,                 (14)
                                                   1 + huT +1 |PT |uT +1 i

where we subtracted B on both sides. We now replace |uT +1 i huT +1 | by its expectation value
                                                                                                  15




FIG. 8: Left: the discrepancies |bT |rms calculated by direct numerical simulation (black line), from
Equation 12 (red dashes), and from Equation 18 (blue dots) for σw = 0.1 mm and no dither. Right:
with 20 µrad round-robin dither added.


and therefore use Q from Equation 11 to arrive at

                                                                 QP̂T
                   b̂T +1 = b̂T ΞT    with      ΞT = 1 −                  ,                   (15)
                                                           1 + Trace QP̂T

where we introduced b̂T = B̂T − B to simplify the writing. Like Equation 12 before is this
a deterministic equation for b̂T that we update in parallel to the stochastic simulations that
generate bT . On the bottom panel in Figure 6 we show |bT |rms , the rms value of bT , as a
solid black line and |b̂T |rms as dashed red line for a simulation with parameters specified in
the figure caption. We take notice that both black and red curves track one another very
well, which allows us to determine the time scales from analyzing ΞT from Equation 15. As
before, we use a coordinate system in which Q and PT are diagonal, ignore the denominator
with the trace, and analyze one mode at a time. If we denote the eigenvalue of ΞT by ξj
(and omit the index j henceforth, because we consider one mode at a time and want to use
the subscript to denote the iteration), we find

                                                 1 − x0 λ + x0 λT
                                ξT = 1 − λxT =                    ,                             (16)
                                                    1 + x0 λT

where we substituted xT from Equation 13. Again the time scales are determined by 1/λ,
the inverse eigenvalues of Q. Inspecting Equation 15, we see that the eigenvalues Ξs describe
how the modes decrease from one iteration T − 1 to T . In order to find the total reduction
                                                                                                           16

after T iterations we need to multiply all the previous eigenvalues ξs for 1 ≤ s ≤ T , which
gives us the eigenvalues yT of the product YT = Ts=1 Ξs
                                                 Q

                                 T            T
                                 Y            Y 1 − x0 λ + x0 λs           1
                          yT =         ξs =                        =             ,                       (17)
                                 s=1          s=1
                                                    1 + x0 λs          1 + x0 λT

where the last equality is straightforward to prove by induction. Thus YT is a diagonal
matrix with expressions 1/(1 + x0 λT ) along its diagonal. If we now rewrite this equation in
non-diagonal coordinates, we obtain the matrix GT = OYT O> that maps the initial b̂0 to b̂T
after iteration T via
                                                                                             
                                                                1                   1
           b̂T = b̂0 GT     with          GT = O diag                    ,...,                    O>     (18)
                                                            1 + x0 λ 1 T       1 + x0 λ m T
and λj = σw2 σj KK > + z 2 /m without iterating through all the intermediate steps. In
                   

passing, we point out that GT behaves like a transfer function that maps the initial b̂0 to
a later value b̂T . Iterating with, for example, different dither amplitudes z only involves
left-multiplying with different GT , each one calculated with the appropriate z.
   Figure 8 shows several discrepancies |bT |rms as a function of the iteration number using
double-logarithmic scales. On the left-hand plot we use a configuration with σw = 0.1 mm
and no dithering. The black line shows |bT |rms from the stochastic simulation, the red line
shows |b̂T |rms using the deterministic iteration, while the blue dots are calculated with the
matrix GT . We observe that all three curves track one another very well. The plot on the
right-hand side in Figure 8 shows the configuration with 20 µrad dithering added, already
used in Figure 6 with the blue dots from the analytic calculation superimposed. Again, the
agreement is rather good, though some discrepancies show up, once |bT |rms becomes very
small. Let us therefore analyze this late regime more carefully.
                                                                                      −1
   From the discussion in Section 2 we know that 2σw2 PT = 2σw2 UT> UT                        is a data-driven
approximation of the covariance matrix for the matrix elements of B̂T . We therefore heuris-
                                             p
tically approximate the error bars by σ(B̂) = 2|PT |rms σw and show |bT |rms for a numerical
simulation (solid black) and the deterministic average (red dashes) as well as σ(B̂) (blue
dash-dots) in Figure 9. We observe that once |bT |rms becomes smaller than σ(B̂) the nu-
merical simulation significantly differs from the averaged model. In this regime the ap-
proximations, in particular, factoring the expectation value of the product of B̂T − B and
|uT +1 i huT +1 | PT into separate expectation values no longer hold. Here, the statistical fluctu-
                                       √
ations around the mean and the 1/ T scaling of the error bars (blue dash-dots) become the
                                                                                               17




FIG. 9: The rms value of the discrepancy |bT |rms from a numerical simulation (solid black) and
from iterating Equations 12 and 15 as well as the approximate error bars σ(B̂) (blue dashes) for a
configuration with σw increased to 0.3 mm and no dither.


dominating factor for the rate of convergence. We therefore need to address the asymptotic
regime separately, which is the topic of the next section.


                                   VI.    ASYMPTOTICS


   The asymptotic regime is characterized by the discrepancy |bT |rms being smaller than
the error bars, or heuristically; the signal |bT |rms is inside the noise floor. We saw in the
simulations shown on the figures that even in this regime B̂T converges towards the ”real”
response matrix B. If we focus on cases without dithering (z = 0), we can explore this
further by exploiting a theorem by Lai and Wei [21], which states that
                                          s                 !
                                            log σmax PT−1
                                                      
                          |B̂T − B|∞ = O                        ,                            (19)
                                               σmin PT−1
                                                        


where σmin [·] and σmax [·] denote the smallest and largest eigenvalue of the matrix in the
argument, respectively. | · |∞ denotes the largest value of the matrix in the argument, which
is always larger than the rms value of all matrix elements that we used in the previous
                                                                                             18

sections; the two values only differ by a numerical factor of order unity. The symbol O(·)
                                                         PT
denotes the leading order in T, and PT−1 = UT> UT =         t=1 |ut i hut | was defined earlier.
                                                PT
We therefore need to determine the scaling of t=1 |ut i hut | and its smallest and largest
eigenvalues with T .
   To do so, we note that the system, defined by Equation 2, can be written as |xt+1 i =
(1 − BK) |xt i + |wt i, which shows that the time step t + 1 only depends on parameters
at time t, which makes it a Markov chain. Moreover, if the closed-loop system is stable,
the spectral radius ρ(Λ), with Λ = 1 − BK, is strictly less than unity, which causes the
process to forget all uniformly bounded initial conditions sufficiently fast. This makes the
corresponding Markov chain uniformly ergodic and implies that the time-average and the
average over the distribution function of the noise, the expectation value E {·}, are the same
                                      T
                               1X
                                     |ut i hut | = E{|uT i huT |} + o(1) ,                 (20)
                               T t=1
where, as before, o(1) is an expression that vanishes in the limit of large T . The right-
hand side of Equation 20 we already calculated in Equation 10 and turn to its asymptotic
                                                     P −1       s
behavior, which is encapsulated in the limit of ΓT = Ts=0   ΛΛ> for large T . First we
note that
                        T −1                    ∞
                        X
                                      > s
                                                X                      1
                                                      ρ(Λ)2s 1 =
                                       
                                ΛΛ          ≤                                1<∞           (21)
                        s=0                     s=0
                                                                   1 − ρ(Λ)2
is finite. Second, the existence can be proven by noting that ΓT is a Cauchy sequence;
ΓT − ΓT 0 = o(1) for large T, T 0 . We can therefore introduce Γ = limT →∞ ΓT and obtain

                                     E{|uT i huT |} = σw2 KΓK > + o(1).                    (22)

This expression allows us to determine the smallest and largest eigenvalue of the left-hand
side
                        σmin [E {|uT i huT |}] = σw2 σmin KΓK > + o(1)
                                                              
                                                                                           (23)

and likewise for σmax . We note that the smallest and largest eigenvalues of a matrix are
continuous functions of the matrix elements. This implies—as a consequence of the contin-
uous mapping theorem [22]—that limits of these functions are preserved, even if the matrix
elements depend on random variables. We therefore obtain from Equation 20
                       " T            #
                        X
                  σmin     |ut i hut | = σmin [T E {|uT i huT |} + o(T )]
                               t=1
                                                  = σw2 T σmin KΓK > + o(T )
                                                                   
                                                                                           (24)
                                                                                                  19




FIG. 10: Left: plotting the logarithm Trace PT> PT versus the log of the number of iterations for
                                                  

σw = 0.1 mm shows that the slope is close to -2. Right: In the same way plotting the log of |bT |rms
shows a slope of approximately κ = −0.57. The red dashes denote the fitted straight lines.


and likewise for σmax . Here o(T ) denotes a quantity that increases strictly slower with T
than T . Moreover, the convergence of the random variables on the left-hand side towards
the expectation value on the right-hand side happens with probability 1—almost surely in
the mathematical literature. Summarily, both σmin PT−1 and σmax PT−1 asymptotically
                                                                     

scale linearly with T .
   For the asymptotic approach of the estimate B̂T towards the “real” response matrix B
we insert the eigenvalues in Equation 19 and find
                                           s                        !
                                                       log T
                            |B̂T − B|∞ = O                              ,                       (25)
                                                  T σmin [KΓK > ]
                                                                                         −1
where we did not spell out constant factors. In passing we note that PT = UT> UT               scales
with 1/σmin PT−1 and this leads to
               

                                                           
                                                  1
                              PT = O                          ,                                 (26)
                                        σw2 T σmin [KΓK > ]

which decreases like 1/T in the leading order.
   In order to verify the asymptotics numerically we run simulations with σw = 0.1 mm for
5 × 106 iterations. Figure 10 shows the asymptotic behavior of Trace PT> PT and of |bT |rms
                                                                           

as a function of the iteration number on a double logarithmic scale in the range between
2.5 and 5 × 106 iterations. A linear fit to the data on the left-hand side shows a slope of
                                                                                                20




FIG. 11: Top row: histograms of the final discrepancy |bf |rms after 5 × 106 iterations (left) with
σw = 0.1 mm, the slope of Trace PT> PT (center) and the slope of |bT |rms (right) in the range
                                       

2.5 × 106 to 5 × 106 iterations. Bottom row: the corresponding plots with 20 µrad round-robin
dither added.

−1.92, if fitting the entire range, and −1.93, if fitting the upper 20 %. This indicates an
approximate tendency towards Trace PT> PT ∝ 1/T 2 , which is consistent with Equation 26.
Repeating these calculations for different random seeds gives comparable results. On the
                                                                             √
other hand, the slope of |bT |rms is approximately 0.57, which is close to 1/ T , the dominant
dependence in Equation 25. But the the curve is much more noisy, which we attribute to
the logarithm of T in the numerator of Equation 25.
   In order to explore this variability we run the simulation with 400 different random
seeds, all having σw = 0.1 mm, and plot the final value of the discrepancy |bf |rms , the slope
of Trace PT> PT , and the slope of |bT |rms in the top row of histograms in Figure 11. We see
                

that after 5 × 106 iterations |bf |rms has reached a value of about 12 mm/rad (left). The slope
of Trace PT> PT is −1.92 (center) and has not quite reached its asymptotic value of −2.
                 

The asymptotic slope of |bT |rms (right) is approximately 0.75. The width of the histograms
                                                                                               21

indicate their standard deviations, which is indicated as the uncertainty in the respective
legends of the plots. We observe that the results are reasonably stable and give a good
indication of the asymptotic behavior of the system. In the bottom row in Figure 11 we
show the corresponding plots for the situation, where 20 µrad round-robin dither is added.
We find that the final value of |bf |rms is only 4 mm/rad (left), while the slope of Trace P > P
                                                                                                 

is very close to the asymptotic value of −2. The slope of the discrepancy |bT |rms (right)
indicates a value of approximately −0.53. We point out that the width of the two histograms
on the right is much larger than the others, which we again attribute to the logarithm in
the numerator of Equation 25.


                          VII.   SOME TECHNICAL ASPECTS


   We now turn to practical aspects of our system to determine the “real” response matrix
B. From Equations 25 and 26 we see that the most important quantity for convergence is
                                                    > s
the smallest eigenvalue of KΓK > , where Γ = ∞
                                            P        
                                             s=0 ΛΛ     is defined immediately before
Equation 22. For all well-behaved feedback systems ρ(Λ) = ρ(1 − BK) is much smaller
than unity and the term with s = 0 dominates the sum, which makes Γ very close to the
m × m unit matrix. Since we do not a priori know B, we just set Γ to the unit matrix when
evaluating the performance of our system and consider KK > alone.
   If the feedback system is equipped with more correctors than position monitors (n < m),
the matrix KK > is degenerate a has a null eigenvalue, which spoils the convergence. The
left-hand plot in Figure 12 shows what happens when we remove one row, corresponding to
one position monitor, from the response matrix and repeat the analysis. The orbit, shown
on the upper panel is still corrected with a rms value comparable to σw , but |bT |rms , shown
on the lower panel, no longer converges to zero. The identification of the response matrix
only works partially and a finite difference to the “real” B remains.
   If, on the other hand, there are more position monitors than corrector magnets (n > m)—
in the simulation we removed one column, corresponding to one corrector magnet, from the
response matrix—the identification of the response matrix works well, as illustrated on the
lower panel on the right-hand plot in Figure 12, because m × m matrix KK > has full rank—
no null eigenvalues. On the other hand, we can no longer correct the orbit, as shown on the
upper panel, because now the m × n matrix K now has eigenvalues null. We can, however,
                                                                                                  22




FIG. 12: Left: one position monitor is removed (n < m), which spoils the convergence of |bT |rms
to zero, but maintains a small orbit. Right: one corrector magnet is removed (n > m); now the
identification of the response matrix works well and |bT |rms converges to zero, but the orbit is not
corrected properly.


remedy this problem by decomposing the symmetric n × n matrix K > K = ODO> , where
D is a diagonal matrix containing the eigenvalues di and O is an orthogonal matrix, whose
                                                                      P
columns are the corresponding eigenvectors |oi i. We note that Φ = i in nullspace |oi i hoi |
is a projection matrix onto the null-space of K > K, such that Ψ = 1 − Φ projects onto its
orthogonal complement, which is the space of BPM readings that the correctors can actually
affect. If we use Ψ |xi instead of |xi when we apply the correction, the null-modes never
pile up and become unstable. If we apply this method to the example from the right-hand
side in Figure 12, the orbit in the upper panel looks very similar to the one on the left-hand
plot. Since we always know K (as opposed to B, which we do not know), we can always
construct Ψ. Using the projector Ψ we can also use our algorithm if there are more BPM
than correctors.
   For one-to-one orbit correction feedback systems with equal number of position monitors
and correctors (n = m) we just have to evaluate the eigenvalues of KK > and possibly adjust
K by hand in order to speed up the convergence, albeit at the expense of compromising the
orbit correction to some extent. The details depend on the particular accelerator and we
will not dwell on this point further.
   In order to understand the scaling of the convergence with system parameters, we consider
                                                                                                  23

rings with increasing number of n = m cells with equal phase advance that contain one
corrector and one BPM, each, which results in a near-circulant response matrix [23]. In
numerical experiments we find that the largest eigenvalue of B > B approximately increases
with n2 . Since the correction matrix K is normally close to the pseudo-inverse of B, we
expect the smallest eigenvalue of KK > to have an inverse dependence on n2 . Moreover, B
is proportional to a typical value of the beta function β̂ in the ring, which makes K ∝ 1/β̂,
                                          2
such that we find σmin KK > ∝ σw /nβ̂ ; the algorithm works best in small rings with
                        

noisy BPM.
   It is instructive to compare the achievable error bars for the response matrix with those
of an open loop measurement campaign, which are approximately given by σ(B)o ≈ σw /θ̂,
where θ̂ is the amplitude of the corrector excitations. In Section II we found that error
bars of B̂ from the closed-loop measurements are given by σ(B̂)2 = diag 2σw2 (UT UT> )−1 =
                                                                                       

2σw diag [PT ]. Moreover, during the early stages of the convergence, the eigenvalues of PT
are given by Equation 13. We see that all eigenvalues decrease with x0 /(1 + x0 λi T ), albeit at
a slow time scale, characterized by the eigenvalues λi of Q. This process continues until the
asymptotic regime is reached, as discussed near the start of Section VI. In the asymptotic
regime PT continues to decrease as specified by Equation 26. We conclude that the error
bars always get smaller and do so without limit. Additionally, Equation 25 implies that the
approximation B̂ asymptotically approaches the “true” response matrix B.
   Finally, extending the algorithm to include settling time τ , processing delay d, and re-
laxation into a new equilibrium with time scale τd is straightforward by introducing unob-
servable state variables |αt i and |βt i. Their dynamic behavior is described by

                τ              1                                 τd                1
    |αt i =        |αt−1 i +      |ut−1−d i   and    |βt i =          |βt−1 i +        B |αt i   (27)
              τ +1           τ +1                              τd + 1           τd + 1

with |ut−1−d i = −K |xt−1−d i. The delay d and time constants τ and τd affect the stability of
the closed-loop system, but we assume that the feedback designer has chosen K to ensure
its stability. In the equation, |αt i corresponds to the field inside the vacuum chamber that
the beam actually “sees” and |βt i, for example, the damping due to synchrotron radiation.
The observable beam position |xt i then updates with |xt+1 i = |xt i + |βt i. We note that the
left side of Equation 27 enables us to uniquely determine the |αt i from the |ut i, which makes
them quasi observable, provided τ and d are known. Moreover, we find the |βt i = |xt+1 i−|xt i
from the |xt i, which turns the right side of Equation 27 into (τd + 1) |βt i − τd |βt−1 i = B |αt i.
                                                                                             24

We observe that this equation has the same form as Equation 3 from the main text with
one component of the left-hand side taking the place of xis+1 − xis shifted by one time step.
Likewise, |αt i takes the place of |us i. The analysis from the report up to Equations 6 and 7
remains valid, but analyzing the convergence and the asymptotics goes beyond the scope of
the present report.


                                  VIII.   CONCLUSIONS


   We applied standard system identification techniques, based on recursive least-squares
methods, to determine the response matrix in parallel to correcting the orbit in a storage ring.
Simulations show that the method works well, though it is rather slow and requires a large
number of iterations. The speed can, however, be increased significantly by systematically
adding small perturbations to the corrector magnets, so-called dithering. In this way a
small deterioration of the orbit quality can be balanced with the desire to determine an
accurate response matrix. We found that the convergence of B̂T to the “real” response
matrix is governed by the eigenvalues λ of the matrix Q from Equation 11 and we solved
the time dependence of the discrepancy b̂T = B̂T − B with some approximations. We found
in Equation 18 that b̂T scales with 1/T , but only until the magnitude of b̂T becomes smaller
                                                               √
than the error bars of the fitting process, which scale with 1/ T . Once inside the noise level,
                                                                 p
we found that the asymptotic behavior of the convergence has a log(T )/T dependence and
is governed by the smallest eigenvalue of KK > . In particular, both the error bars of the
approximation B̂T and the difference between B̂T and the “real” B tend to zero in the limit
of large T . Furthermore, we found that those feedback systems with number of BPMs equal
or larger than the number of correctors (n ≥ m) permit us to simultaneously stabilize the
orbit and to identify the response matrix B.
   Several extensions of this work come to mind. First, optimizing the correction matrix
K such that the smallest eigenvalue of KK > is as large as possible without spoiling the
orbit quality σx . Second, comparing different correction strategies, for example, deriving K
from “optimal control” quality measures that put a weight both on the orbit σx2 and the
rms corrector excitation. Third, finding an optimal strategy to make the dither amplitude
z time-varying, such that global measure of performance that balances orbit correction and
system identification is minimized. The regret, studied for instance in [24], may serve as an
                                                                                                       25

example.


                                          Acknowledements


   This work was supported in part by the Swedish Research Council (grant 2016-00861),
and the Swedish Foundation for Strategic Research (Project CLAS).


                           Appendix A: Sherman-Morrison formula


   Here we show that PT +1 is given by Equation 6 if its inverse PT−1               −1
                                                                    +1 is given by PT +1 =

PT−1 + |uT +1 i huT +1 |. To show this, we explicitely calculate PT−1
                                                                    +1 PT +1 and show that it

evaluates to the unit matrix
                                                                                  
          −1
                       −1                                PT |uT +1 i huT +1 | PT
        PT +1 PT +1 = PT + |uT +1 i huT +1 | PT −                                                    (A1)
                                                           1 + huT +1 |PT |uT +1 i
                        −1
                    = PT PT + |uT +1 i huT +1 | PT
                        P −1 PT |uT +1 i huT +1 | PT + |uT +1 i huT +1 | PT |uT +1 i huT +1 | PT
                      − T
                                                 1 + huT +1 |PT |uT +1 i
                                                   |uT +1 i (1 + huT +1 |PT |uT +1 i) huT +1 | PT
                    = 1 + |uT +1 i huT +1 | PT −
                                                               1 + huT +1 |PT |uT +1 i
                    = 1,

and we can use Equation 6 to update PT with the new information that is encoded in the
new corrector excitations |uiT +1 . Note that PT = (UT> UT )−1 and its inverse are symmetric
by construction for all T . This implies that the order of multiplication of PT +1 and its
inverse does not matter and we also have PT +1 PT−1
                                                  +1 = 1.




                             Appendix B: Response-matrix update


   Here we follow [13] and show that the update of the response matrix B̂ is accomplished
by Equation 7. We therefore write Equation 5 for time step T + 1
                                                   
                                                 i
                                               y
                                           1 
                                                               " T                               #
                                 −1             .                X
      |B̂Ti:+1 i = UT>+1 UT +1      UT>+1  ..  = PT +1              ysi |us i + yTi +1 |uT +1 i
                                                  
                                                                s=1
                                             yTi +1
                                                   "X
                                                      T
                                                                                    #
                          PT |uT +1 i huT +1 | PT
                 = PT −                                 ysi |us i + yTi +1 |uT +1 i .                (B1)
                          1 + huT +1 |PT |uT +1 i s=1
                                                                                                             26

Here we introduce the abbreviation yTi = xiT +1 − xiT , exploit that UT>+1 = (|u1 i , . . . , |uT +1 i),
and finally express PT +1 through Equation 6. In the next step we multiply the two square
brackets and obtain four terms
                            T
                                                               PT |uT +1 i huT +1 | PT Ts=1 ysi |us i
                            X                                                         P
        |B̂Ti:+1 i   =   PT      i             i
                                ys |us i + PT yT +1 |uT +1 i −
                            s=1
                                                                     1 + huT +1 |PT |uT +1 i
                                  PT |uT +1 i huT +1 | PT |uT +1 i yTi +1
                              −
                                        1 + huT +1 |PT |uT +1 i
                            i:        i               PT |uT +1 i huT +1 |B̂Ti: i
                     =   |B̂T i + PT yT +1 |uT +1 i −                                                       (B2)
                                                      1 + huT +1 |PT |uT +1 i
                                        PT |uT +1 i huT +1 | PT |uT +1 i
                              −yTi +1
                                           1 + huT +1 |PT |uT +1 i
where, according to Equation 5, we identify the estimate in the previous iteration T as
|B̂Ti: i = PT Ts=1 |us i ysi . Combining the second and the fourth term, we arrive at
             P

                                        h                                   i        PT |uT +1 i
              |B̂Ti:+1 i = |B̂Ti: i +       xiT +2 − xiT +1 − huT +1 |B̂Ti: i                           .   (B3)
                                                                                1 + huT +1 |PT |uT +1 i
Taking the transpose of this equation and stacking the rows on top of each other then leads
to Equation 7.


                                   Appendix C: Code for one iteration


   The following function receives B̂ and P , as well as the recently measured orbit |xi and
the dither vector |zi as input and returns the updated matrices B̂new and Pnew as well
as the orbit |xnew i after the correction is applied. Inside the function, first the externally
defined noise level σw , a constant correction matrix B̃, the “real” response matrix B, and
the correction matrix K are supplied as global variables. Next, using K, the new corrector
values |ui are calculated, dither |zi is added to the correctors, and the new orbit |xnew i
is calculated, including the noise |wi, here implemented as normally distributed random
numbers. Then the auxiliary quantity huT +1 | PT is stored in the variable tmp and the inverse
of the denominator in the last term in Equation 6 is calculated. The next two lines are
straight implementations of Equations 6 and 7.

  function [Bhatnew,Pnew,xnew]=one_iteration4(Bhat,P,x,z)
  global sig Btilde Breal Bplus % noise, est., real, corr.
  %   u=-Bhat\x+z; % adaptive feedback
                                                                                              27

  u=-Bplus*x+z;                                             % eq. 2 + dither
  xnew=x+Breal*u+sig*randn(size(x));                        % eq. 2
  tmp=u’*P;                                                 % <u|P
  denominv=1/(1+tmp*u);                                     % 1/(1+<u|P|u>)
  Pnew=P-tmp’*tmp*denominv;                                 % eq. 6
  Bhatnew=Bhat+(xnew-x-Bhat*u)*tmp*denominv; % eq. 7

The figures in the main body of the report are produced by iterating this function. Note
that in the above code the correction matrix K is fixed. We can, however, easily make the
feedback adaptive by simply replacing this line in the code by u=-Bhat\x+z, as indicated in
the commented-out line. In this way, always the most recent approximation for the matrix
B̂T is used when correcting the orbit.


                     Appendix D: Spatially correlated monitor noise


   The assuption that the noise |wt i of position monitors is uncorrelated, is easily relaxed and
in this appendix we show spatially correlated noise, characterized by E {|vt i hvs |} = σw2 δts C
affects the rest of the results, where σw2 C is the covariance matrix of the noise. Its matrix
elements on the diagonal σw2 Cii describe the square of the error bars of BPM i and the off-
diagonal elements describe the correlations among different BPMs. Note that we separated
the magnitude of the noise (σw2 ) from the correlations, where C is a positive definite and
symmetric matrix with matrix elements of order unity.
   Since C is symmetric we can decompose it into orthogonal matrices O and a diagonal
matrix. Since it is positive definite, all its eigenvalues are positive and we can write the
diagonal matrix as the square of another diagonal matrix D

                                         C = OD2 O> .                                       (D1)

We will now use this representation of C to transform the dynamical system represented by
Equation 2, but with correlated noise |vt i

                                 |xt+1 i = |xt i + B |ut i + |vt i .                        (D2)

and multiply it with D−1 O> from the left, which results in

                D−1 O> |xt+1 i = D−1 O> |xt i + D−1 O> B |ut i + D−1 O> |vt i .             (D3)
                                                                                                         28

With the transformed variables

               |yt i = D−1 O> |xt i ,         B 0 = D−1 O> B,                 |wt i = D−1 O> |vt i .   (D4)

Equation D3 reads
                                   |yt+1 i = |yt i + B 0 |ut i + |wt i ,                               (D5)

where we have

     E {|wt i hws |} = D−1 O> E {|vt i hvs |} OD−1 = D−1 O> σw2 Cδts OD−1 = σw2 δts 1 .                (D6)

We find that this system is equivalent to the one from Equation 2, such that we can directly
use the methods developed in the main body of this report. We only need to undo the
transformation from Equation D4 in the end.
   If we apply this procedure to Equation 6 and 7 we find that these equations are unchanged;
the improvement of the B̂ does not depend on the noise as long as there are perturbations.
Only the changes of the controller |ut i and the resulting orbit changes matter.
   The correlation matrix C does, however, affect the convergence of the algorithm. Using
correlated noise |vt i instead of |wt i in Equation 10, we find that its last equality becomes
                                                        T −1
                                                        X                s
                        E{|uT i huT |} = σw2 K                 Λs C Λ>        K > + o(1) .             (D7)
                                                        s=0

Following the reasoning from the main body, the term with s = 0 is dominant, which gives us
                                                                                                        2
E{|uT i huT |} = σw2 KCK > and the matrix Q from Equation 11 becomes Q = σw2 KCK > + zm 1.
With this version of Q the conclusions of Section V remain the same.
   Also the asymptotic behavior is affected by C. Equation 21 becomes
                        T −1                      ∞
                        X
                               s        > s
                                                  X                              ρ(C)
                                                        ρ(C)ρ(Λ)2s 1 =
                                         
                 ΓT =          ΛC Λ           ≤                                          1<∞           (D8)
                        s=0                       s=0
                                                                               1 − ρ(Λ)2

where ρ(C) is the spectral radius of C. This results in a slight redefinition of Γ = limT →∞ ΓT
which still is finite, which renders the remainder of the Section VI valid.




 [1] J. P. Koutchouk, Trajectory and closed orbit correction, in Frontiers of particle beams, Springer
    Lecture Notes in Physics 343 (1989) 46.
                                                                                                 29

 [2] M. Minty, F. Zimmermann, Measurement and Control of Charged Particle Beams, Springer,
     Heidelberg, 2003.
 [3] X. Huang, Beam-based correction and optimization for accelerators, CRC press, Boca Raton,
     2020.
 [4] V. Ziemann, Imperfections and correction, contribution to the CERN Accelerator School:
     Introduction to Accelerator Physics, https://arxiv.org/abs/2006.11016, June 2020.
 [5] G. Rehm, Characterization of closed orbit feedback systems, Proceedings of the eighth beam
     instrumentation conference IBIC2019 in Malmö, 2019, p. 479.
 [6] S.H. Mirza, R. Singh, P. Forck, B. Lorentz, Performance of the closed orbit feedback systems
     with spatial mismatch, Physical Review Accelerator and Beams 23 (2020) 072801.
 [7] J. Corbett, M. Lee, V. Ziemann, A Fast Model Calibration Procedure for Storage Rings,
     Proceedings of the Particle Accelerator Conference PAC93, Washington, 1993, p. 108.
 [8] J. Safranek, Experimental determination of storage ring optics using orbit response measure-
     ments, Nuclear Instruments and Methods A 388 (1997) 27.
 [9] W. Corbett, J. Safranek, D. Robin, V. Ziemann, Debugging real accelerators, Particle Accel-
     erators 58 (1997) 193.
[10] G. Goodwin, R. Payne, Dynamic System Identification, Academic Press, London, 1977.
[11] L. Ljung, System Identification; theory for the user, 2nd ed., Prentice Hall, New Jersey, 1999.
[12] W. Press et al., Numerical Recipes, 2nd ed., Cambridge University Press, Cambridge, 1992.
[13] Section 7.2 in [10].
[14] V. Ziemann, Hands-on accelerator physics using Matlab, CRC press, Boca Raton, 2019.
[15] I. Martin et al., A Fast Optics Correction for the Diamond Storage Ring, Presented at the
     International Particle Accelerator Conference IPAC2014 (2014) 1763.
[16] X. Yang, V. Smaluk, L. H. Yu, Y. Tian, K. Ha, Fast and precise technique for magnet lattice
     correction via sine-wave excitation of fast correctors, Physical Review Accelerator and Beams
     20 (2017) 054001.
[17] Z. Marti et al., Fast Orbit Response Matrix Measurements at ALBA, Presented at the Inter-
     national Particle Accelerator Conference IPAC2017 (2017) 365.
[18] M. Ross, L. Hendricksen, T. Himel, E. Miller, Precise system stabilization at SLC using dither
     techniques, SLAC-PUB-6102, 1993, presented at the Particle Accelerator Conference (PAC93)
     in Washington, D.C.
                                                                                                 30

[19] S. Gierman et al., New fast dither system for PEP-II, SLAC-PUB-12679, presented at the
    European Particle Accelerator Conference (EPAC06) in Edinburgh, Scotland.
[20] M. Masuzawa et al., Early commissioning of the luminosity dither system for SuperKEKB,
    presented at the seventh International Beam Instrumentation Conference (IBIC18) in Shang-
    hai, China, 2018.
[21] T. Lai, C. Wei, Least squares estimates in stochastic regression models with applications to
    identification and control of dynamic systems, The Annals of Statistics 10 (1982) 143.
[22] Van der Waart, Asymptotic Statistics, Cambridge University Press, 1998; see Theorem 2.3.
[23] S. Mirza, R. Singh, P. Forck, H. Klingbeil, Closed orbit correction at synchrotrons for symmet-
    ric and near-symmetric lattices, Physical Review Accelerator and Beams 22 (2019) 072804.
[24] I. Ziemann, H. Sandberg, On Uninformative Optimal Policies in Adaptive LQR with Unknown
    B-Matrix, arXiv:2011.09288, https://arxiv.org/abs/2011.09288.
