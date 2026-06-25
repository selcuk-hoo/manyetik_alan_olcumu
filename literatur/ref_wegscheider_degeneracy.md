# REFERANS: Wegscheider et al., Inverse modeling of circular lattices via orbit response in the presence of degeneracy (PRAB 26, 032803, 2023)

> pdftotext, özgünlük karşılaştırması (claim B)

---

                   PHYSICAL REVIEW ACCELERATORS AND BEAMS 26, 032803 (2023)



                   Inverse modeling of circular lattices via orbit response measurements
                                      in the presence of degeneracy
                                                                D. Vilsmeier*
                           Johann Wolfgang Goethe-University Frankfurt, 60323 Frankfurt am Main, Germany

                                                                  R. Singh
                       GSI Helmholtz Centre for Heavy Ion Research, 64291 Darmstadt, Planckstr. 1, Germany

                                                                   M. Bai
                  SLAC National Accelerator Laboratory, 2575 Sand Hill Road, Menlo Park, California 94025, USA

                             (Received 27 October 2022; accepted 10 February 2023; published 27 March 2023)

                   The number and location of beam position monitors (BPMs) and steerers with respect to the quadrupoles
                in a circular lattice can lead to degeneracy in the context of fitting linear optics and extracting lattice
                information from measured closed orbits. Furthermore, the measurement uncertainties due to the
                imperfection of BPMs and steerers can be propagated by the fitting process in ways that prohibit the
                successful extraction of discrepancies between lattice elements in the real machine and their description in
                the corresponding model. We systematically studied the influence of the placement of BPMs and steerers
                on the reconstruction of linear optics and corresponding lattice information. The derivative of orbit
                response coefficients with respect to the quadrupole strengths, the Jacobian, is derived as an analytical
                formula. This analytical version of the Jacobian is used to further derive the theoretical limitations of fitting
                linear optics from closed orbits in terms of the placement of BPMs and steerers. It is further demonstrated
                that when evaluating the Jacobian during the fitting procedure, the analytical version can be used in place of
                the conventional finite-difference computation. This allows for greatly improved efficiency when
                computing the Jacobian during each iteration of the fitting procedure. The approach is tested with
                large-scale simulations and the findings are verified by measurement data taken on SIS18 synchrotron at
                GSI Helmholtz Centre for Heavy Ion Research. The presented methods are of general nature and can be
                applied to other accelerator lattices as well. The fitting procedure by using the analytical Jacobian is tested
                in conjunction with various methods for mitigating quasidegeneracy and the results agree with those
                obtained by using the conventional Jacobian via finite-difference approximation.

                DOI: 10.1103/PhysRevAccelBeams.26.032803



                      I. INTRODUCTION                                      those properties from measuring their effect on the beam
                                                                           behavior have been employed in order to improve the
   Precise knowledge of the lattice’s optics elements is
                                                                           quality of the accelerator model with respect to the real
crucial for the optimal operation of any circular accelerator.
                                                                           machine. The goal of deriving those properties constitutes
Inability to identify or counteract discrepancies between the
                                                                           an inverse problem since the observed beam behavior is
lattice elements in the real machine and their description in
                                                                           used to estimate the values of the underlying lattice
the corresponding model, also known as model errors, in
                                                                           parameters which gave rise to this behavior. Solving such
general, can result in failure to preserve beam parameters or
                                                                           an inverse problem is generally referred to as inverse
hinder further improvements of the machine performance.                    modeling. In this context, extracting linear optics and
Since the physics properties of lattice elements such as                   model errors from closed orbit measurements is a typical
quadrupole gradients are rather difficult to be precisely                  method for solving the inverse problem in terms of the
measured directly in the machine, techniques for deriving                  quadrupole contributions in the lattice. This method
                                                                           requires a measured orbit response matrix (ORM) as input
  *
      d.vilsmeier@gsi.de                                                   and then varies all relevant lattice parameters in a multi-
                                                                           dimensional optimization problem to match the simulated
Published by the American Physical Society under the terms of              with the measured ORM. Based on the outcome of the
the Creative Commons Attribution 4.0 International license.
Further distribution of this work must maintain attribution to             optimization procedure, model parameters are adjusted and
the author(s) and the published article’s title, journal citation,         the adjusted model is expected to accurately represent the
and DOI.                                                                   actual machine. The first detailed discussion on this method

2469-9888=23=26(3)=032803(21)                                     032803-1                Published by the American Physical Society
D. VILSMEIER, R. SINGH, and M. BAI                                     PHYS. REV. ACCEL. BEAMS 26, 032803 (2023)

can be found in [1] and since then the technique has been        minimization problem. The matrix that is obtained by
further improved and has experienced frequent usage at           concatenating the derivatives for each lattice parameter
different institutes [2–5]. The method has been implemented      as column vectors is generally referred to as the Jacobian
as a Matlab program called LOCO which is part of the Matlab      matrix. In this contribution, we derive an analytical
Middle Layer (MML) for Accelerator Control [6]. Because          version of the Jacobian relating the ORM to the quad-
measurement of the ORM typically varies one steerer at a         rupole strength errors along with BPM and steerer gain
time, it can take a significant amount of machine time. There    errors. This Jacobian matrix is used by the optimizer, e.g.
have been efforts to reduce the time and impact of the           Levenberg-Marquardt, in order to improve the current best
measurement, for example, by sine-wave excitation of             guess of lattice parameters during an iterative process. We
multiple steerers at different frequencies simultaneously        have studied the properties of this analytical Jacobian with
[7]. Another approach used the data obtained from closed-        respect to the conditioning of the inverse problem. We show
orbit feedback correction to continuously update an estimate     that the analytical Jacobian highlights all relevant proper-
of the ORM; for a sufficient number of iterations, this will     ties of the underlying model error estimation problem.
converge to the true ORM [8]. A recent method proposes to        Rank deficiency of the Jacobian implies a degeneracy of the
use two steerers in each transverse plane to modulate the        inverse problem while small eigenvalues of the Jacobian
closed orbit in an appropriate pattern which also allows for
                                                                 suggest quasidegeneracy for some patterns of quadrupole
reduced measurement times [9].
                                                                 errors. These patterns are more susceptible to the propa-
   Even when the adjusted model correctly reproduces the
                                                                 gation of measurement uncertainty. We further use the
measured data such as ORM, the lattice element parameters
                                                                 analytical version of the Jacobian, obtained from the
in the adjusted model may not necessarily converge to the
corresponding actual physics properties in the real machine      lattice’s Twiss data, during the fitting procedure and show
due to various contributing factors such as the errors of beam   that it reaches convergence similar to using the numerical
position monitors (BPMs) which cast an uncertainty on the        Jacobian which is obtained via finite-difference approxi-
measured ORM [10,11]. This uncertainty then propagates           mation. The analytical Jacobian is obtained quickly since it
through the inverse modeling process and influences the          requires only a single Twiss computation for the lattice.
precision of derived parameters. Depending on the lattice and    This leads to a substantial speed-up factor compared to
the optics, the effect of BPM errors can be more or less         computing the numerical Jacobian via finite-difference
problematic for the accuracy of inverse modeling results. In     approximation. Especially for larger lattices, this greatly
some cases, the influence of BPM errors can even hinder the      improves the compute time for the Jacobian during each
successful reconstruction of quadrupole errors. An improve-      iteration of the fitting procedure and, thus, reduces the time
ment of the efficiency was introduced in [10] by adding          until results are available. In this process, we also devised a
specific constraints for the fitting parameters. A related       general iterative method for automatic and online correc-
approach for improving the efficiency was introduced             tion of quadrupole errors simply based on the analytical
in [11].                                                         Jacobian and measured ORM. This method has similarities
   The property of the lattice that is responsible for the       with iterative closed orbit correction.
propagation of measurement uncertainty during inverse                The presented methods are extensively tested via large-
modeling strongly depends on the availability and location       scale simulations and they are further verified via dedicated
of BPMs and steerers in the lattice as these devices produce     measurements conducted at the SIS18 heavy-ion synchro-
and determine the measured data. While synchrotron light         tron at GSI Helmholtz Centre for Heavy Ion Research. The
sources typically have many BPMs installed, the available        fitting procedure by using the analytical Jacobian has been
BPMs at hadron synchrotrons are rather limited and are           tested in conjunction with various methods for mitigating the
typically not dual plane BPMs. Thus, it becomes increas-         influence of quasidegeneracy [11] and the results agree with
ingly important to understand this relationship. A lack of       those obtained by using conventional numerical Jacobian.
BPMs enhances the degree to which measurement uncer-                 In the following, the structure of the paper is described.
tainty is propagated and might cause specific lattice            In Sec. II, we introduce the lattice which is used throughout
parameters to be especially susceptible to this effect.          this contribution and the concept of the orbit response
This property is referred to as quasidegeneracy. In some         matrix. Section III explains the inverse problem of esti-
situations, the lack of BPMs might even cause the inverse        mating quadrupole errors with regard to the degeneracy
problem to be ill-posed in such a way that the estimated         of its solutions. The analytical derivation of the Jacobian
quadrupole gradients are not uniquely determined by the          is presented. Also, the influence of BPM and steerer
measured ORM data. Thus, it is important to study the            placement on degeneracy and quasidegeneracy is shown.
limitations and properties of the inverse modeling process       Section IV discusses the fitting procedure by using the
with regard to the placement of BPMs and steerers. The           Jacobian as well as discusses the convergence properties
derivatives of the ORM elements with respect to the              for different approaches. This includes the usage of the
relevant lattice parameters, including the quadrupole gra-       analytical Jacobian during the fitting procedure. In Sec. V, the
dients, contain lots of information about the corresponding      experimental results are presented. The fitting procedure,

                                                           032803-2
INVERSE MODELING OF CIRCULAR LATTICES …                                   PHYS. REV. ACCEL. BEAMS 26, 032803 (2023)

including the analytical Jacobian approach, has been imple-
mented as part of a self-developed PYTHON package [12].

           II. ORBIT RESPONSE MATRIX
   The orbit change xb at BPM b when changing the
steerers indexed with s by a kick δs , is given by [13]:
             " pﬃﬃﬃﬃﬃﬃﬃﬃﬃ                                   #
      X           βb βs                             Db Ds
xb ¼      δs              cosðπQ − jμb − μs jÞ −         
       s
              2 sinðπQÞ                            1    1
                                                    2 − 2 C
                                                  γ    γt

                                                            ð1Þ ;

where βb;s and μb;s denote, respectively, the beta functions        FIG. 1. Schematic of SIS18 lattice with optics functions
                                                                    showing the first of in total 12 sections. The 12 sections are
and the phase advances at BPM and steerer position, and Q
                                                                    identical except that in sections 4 and 6, the horizontal steerer is
is the betatron tune. In the second term, Db;s denotes              located on the second bending magnet rather than the first. Blue
the dispersion at BPM and steerer position and C is the             (raised): focusing quadrupoles, red (lowered): defocusing quad-
circumference of the synchrotron; γ and γ t denote, respec-         rupoles, yellow (centered): bending magnets; the horizontal
tively, the beam energy (E) and transition energy of the            steerer is shown as a black line on top of the first bending
lattice (γ ¼ EE0 where E0 is the rest energy of the beam).          magnet (in sections 4 and 6, it is located on the second bending
The energy-dependent term is only relevant for synchro-             magnet); the vertical steerer is shown as a gray box between the
                                                                    focusing and defocusing quadrupole; the vertical and horizontal
trons operating near transition energy.                             BPMs (in that order) are shown as gray solid boxes downstream
   Hence, the orbit change is a linear function in the applied      of the third quadrupole (since they are right next to each other,
kick and it encodes the optics via the lattice functions β and      they might appear as a single gray box).
μ. The orbit response rbs at BPM b reacting to a single
steerer s is defined as
                                                                    (iv) D quads from even numbered sections, and (v) T quads
                              x                                     from all sections.
                         rbs ¼ b :                          ð2Þ
                              δs                                       Each section contains two bending magnets next to each
                                                                    other. The horizontal steerers are placed on the first bending
   The orbit response matrix (ORM) arranges the orbit               magnet, except in sections 4 and 6 where they are placed on
responses for all BPM/steerer pairs in a matrix form: rbs ,         the second bending magnet. The vertical steerers are placed
where b is the row index and refers to BPMs and s is the            between the F and the D quadrupole identically in each of
column index and refers to steerers.                                the sections. The vertical and horizontal BPMs are placed
   The lattice of the rapid cycling synchrotron SIS18 at GSI        downstream of the T quadrupole, identically in each of the
Helmholtz Centre for Heavy Ion Research is used exem-               sections.
plary throughout this contribution. In the future, SIS18 will          Each individual electrode of the “shoebox” type capaci-
serve as the injector for the SIS100 synchrotron which is           tive pickup structure is terminated with 50-ohm amplifiers
part of the FAIR project [14]. This booster operation at very       which is followed by direct digitization at 125 MSa=s. The
high intensities puts stringent requirements on the optics          orbit is calculated by least squares fitting the oppo-
and, thus, a thorough understanding of the linear optics            site electrode signals on a user-defined time window. A
builds the foundation for any further improvements [15].            detailed discussion of the orbit measurement scheme along
The lattice of SIS18 consists of 12 sections. An overview is        with measurement uncertainty estimates can be found
presented in Fig. 1. Each section contains three quadru-            in [16].
poles, labeled F, D, and T, and the placement and strength             The nominal ORM of SIS18 shows a circulant structure
of these quadrupoles are identical in each of the sections.         in the vertical block due to the symmetric placement of
This triplet structure is utilized to increase the transverse       quadrupoles, vertical steerers, and BPMs within each
acceptance during beam injection. The strength of T                 section. Here, circulant means that each column of the
quadrupoles is gradually decreased by 1 order of magnitude          matrix is shifted by one element compared to the previous
during the ramp, resulting in a small strength during               column. Thus, the entire information of a circulant matrix is
extraction optics. The 36 quadrupoles are connected                 encoded in a single column and in the fact that the matrix is
with five distinct power supplies, separating the quadru-           circulant, of course. In the horizontal block, the circulant
poles into the following families: (i) F quads from odd             structure is broken in the two sections 4 and 6 because, in
numbered sections, (ii) F quads from even numbered                  those sections, the horizontal steerer is placed on the second
sections, (iii) D quads from odd numbered sections,                 bending magnet rather than the first.

                                                            032803-3
D. VILSMEIER, R. SINGH, and M. BAI                                           PHYS. REV. ACCEL. BEAMS 26, 032803 (2023)

   The SIS18 lattice will be used for explaining various              an overall more efficient estimator (reducing the mean
important concepts throughout this contribution. Never-               squared error of its predictions).
theless, these concepts are of general nature and are                     The first mention of quasidegeneracy for linear optics
applicable to any other accelerator lattice, too.                     from closed orbits was made in [10]. The proposed solution
                                                                      was to switch from an unbiased to a biased estimator in
                                                                      order to improve the overall efficiency of the estimates.
                    III. DEGENERACY
                                                                      This was done by augmenting the cost function with terms
   The goal of inverse modeling is to minimize the                    that correspond to the various specific quasidegeneracy
disagreement between measured and simulated observ-                   patterns of the lattice parameters. A related approach [11]
ables. The amount of disagreement is quantified by the                limited the change of lattice parameters during each
cost function. Typically, the cost function is given as the           iteration of the optimization by using a dedicated set of
chi-squared weighted sum of squared deviations:                       weights in the cost function.
                                                                          Regarding the terminology, we distinguish between
                           X ðmi − oi Þ2                              (pure) degeneracy and quasidegeneracy. A purely degen-
                    χ2 ¼                   ;                  ð3Þ
                            i
                                    σ 2i                              erate case is one for which there exist multiple distinct
                                                                      solutions that yield the same values for the chosen set of
where oi and σ i are, respectively, the ith observation and           observables in the absence of measurement uncertainty.
measurement uncertainty and mi is the corresponding                   This is the case if, for example, there are too few BPMs
simulated quantity obtained from the model. The vector                available compared to the number of quadrupoles. A
of residuals is defined as r ≡ m − o.                                 quasidegenerate case, on the other hand, is one where
   Any procedure with the goal of predicting a set of model           there exist multiple solutions that are plausible in view of
parameters P which minimize this cost function is referred            the measurement uncertainty, i.e., which can be plausibly
to as an estimator. The efficiency of an estimator can be             explained by the measured data, and some (combinations
quantified by the spread of its predictions around the true           of) parameters are noticeably more susceptible to the effect
parameter values. Thus, the mean squared error (mse)                  of measurement uncertainty than others. The presence of
criterion serves as a measure of estimator efficiency:                measurement uncertainty does not change the nature of the
                                                                      optimization problem though, as there is still a unique
                                                                      global minimum, depending on the specific data used for
      mse½P ¼ E½ðP − θÞ2  ¼ Var½P þ ðE½P − θÞ2            ð4Þ     fitting. Rather, the quasidegeneracy is a property of the
                                                                      modeled system. Depending on the lattice and optics, some
Here, P denotes the predicted parameter values by the                 directions in parameter space cause less increase in the cost
estimator, θ is the true parameter value, and E½· and Var½·         function than others and, thus, are more susceptible to
denote, respectively, the expectation value and the variance          measurement uncertainty. This is sketched in Fig. 2 where
of its argument. The second term in Eq. (4) corresponds to            the orbit response of a single BPM/steerer pair is shown
the bias of the estimator. Thus, regarding the efficiency of          in dependence on the three different types of quadrupoles
an estimator, there is a trade-off between its variance and           of the SIS18 lattice, F, D, and T quadrupoles. Clearly, the
bias, and an increase in the estimator’s bias might result in         change in orbit response is more flat for the T quadrupole




FIG. 2. Example for the orbit response change of a single BPM/steerer pair when varying a single quadrupole. The horizontal orange
area indicates an orbit response uncertainty of 10 μm mrad−1 and is the same for all quadrupoles. The vertical orange area indicates the
corresponding plausible region of the quadrupoles’ K 1 L strengths. Clearly, the plausible K 1 L region is different for the various
quadrupoles and it depends on the steepness of the orbit response change with K 1 L for each quadrupole.


                                                              032803-4
INVERSE MODELING OF CIRCULAR LATTICES …                                   PHYS. REV. ACCEL. BEAMS 26, 032803 (2023)

than for the other two. This example shows only a single            [labeled (B)], the cosine terms still expand into
ORM element, so for the actual optimization problem, the            sinð2μk Þ; cosð2μk Þ while the integral term expands into
situation is more complex but the principle is the same: flat       sinðμk Þ2 ; cosðμk Þ2 ; sinðμk Þ cosðμk Þ terms. By using the
directions in the parameter space are more susceptible to           trigonometric identities sinð2μk Þ ¼ 2 sinðμk Þ cosðμk Þ and
measurement uncertainty. These directions are determined            cosð2μk Þ ¼ cosðμk Þ2 − sinðμk Þ2 , as well as the trigonometric
by the underlying model, i.e., the lattice and optics.              identity 1 ¼ cosðμk Þ2 þ sinðμk Þ2 for the terms that are
                                                                    independent of μk , one can rewrite the whole Eq. (6) in
         A. Analytical derivative of orbit response                 terms of sinðμk Þ2 ; cosðμk Þ2 ; sinðμk Þ cosðμk Þ where the coef-
                                                                    ficients for these terms only depend on μb , μs , and Q. We do
   In order to explain the degeneracy properties of a given
                                                                    not spell out this expanded form of the Jacobian here because
lattice, we consider the orbit response formula rbs for a
                                                                    it is lengthy and it varies across the three distinct cases (A, B,
single dipolar kick and calculate the derivative rkbs with
                                                                    C). However, an overview of the grouped coefficients is
respect to a change in the kth quadrupole’s strength. In the
                                                                    given in the Appendix (Table III). In the following, we focus
following, we assume that the operation is not close to
                                                                    on the following more general observations. Given that the
transition energy and, thus, the energy-dependent term in
                                                                    Jacobian for each BPM/steerer/quadrupole triple can be
Eq. (1) can be neglected.
                                                                    written as the sum of three expressions involving μk [namely,
                    pﬃﬃﬃﬃﬃﬃﬃﬃﬃ                                      sinðμk Þ2 ; cosðμk Þ2 ; sinðμk Þ cosðμk Þ] together with their
                      βs βb
            rbs ¼              cosðπQ − jμs − μb jÞ          ð5Þ    coefficients that depend solely on μb , μs , Q, each column
                  2 sinðπQÞ                                         of the Jacobian can be written as a linear combination
where b and s indicate, respectively, the BPM and steerer           of v1 sinðμk Þ2 þ v2 cosðμk Þ2 þ v3 sinðμk Þ cosðμk Þ where
                                                                    the column vectors v1;2;3 contain the row-wise constant
index. Taking the derivative with respect to the integrated
strength ðK 1 LÞk of the kth quadrupole, we obtain                  coefficients depending only on μb , μs , and Q. The expres-
                                                                    sions for these coefficients are the same for each group of
        drbs                                                        quadrupoles that is not interleaved by BPMs nor steerers.
rkbs ≡                                                              Thus, the column span of the Jacobian is given by the three
      dðK 1 LÞk
                                                                   column vectors v1;2;3 for each group of quadrupoles and,
            βk        1       tanðπQ − jμb − μs jÞ                  thus, for a lattice with N sections and three or more non-
    ¼ −rbs                 þ
            2 2 tanðπQÞ               2                             interleaved quadrupoles per section, the rank of the Jacobian
        cosð2πQ − 2jμb − μk jÞ þ cosð2πQ − 2jμs − μk jÞ             is at most 3N. It should be emphasized that this holds only if
      þ                                                             all the involved quadrupoles in each section are consecutive,
                           2 sinð2πQÞ
                                                                    i.e., not interleaved by BPMs nor steerers since otherwise
        tanðπQ − jμb − μs jÞ                                        their coefficients would change according to the cases (A, B,
      −
               sinð2πQÞ                                             C). This implies that four or more consecutive quadrupoles
        Z maxðμ ;μ Þ                      
                 b s                                                per section will cause a pure degeneracy since their
      ×              cosð2πQ − jμk − ujÞdu ;        ð6Þ             contributions to the Jacobian can still be described by only
            minðμb ;μs Þ
                                                                    three column vectors. This result holds for one dimension
where βk and μk are, respectively, the beta function and            (horizontal or vertical) but for uncoupled optics, it is easily
phase advance at the kth quadrupole. The full derivation is         extended to both dimensions by considering that there are
given in Appendix A.                                                sinðμk Þ2 ; cosðμk Þ2 ; sinðμk Þ cosðμk Þ terms for both dimen-
                                                                    sions separately, i.e., six independent coefficient vectors
                                                                    v1;2;3;4;5;6 . Thus, the dimension of the column span of the
                           B. Pure degeneracy                       Jacobian involving both dimensions is bounded by 6N and,
    A pure degeneracy exists if there is a set of quadrupoles       therefore, seven or more consecutive quadrupoles will cause
that can assume different strengths and this is not reflected in    a pure degeneracy.
the selected observables. Using the ORM as observable, this             This is in agreement with the result derived in [17] which
is the case if there are specific lattice segments of quadrupoles   is that for uncoupled transverse optics, a set of seven or
without BPMs nor steerers in between. By considering                more consecutive quadrupoles in both dimensions (or
Eq. (6) together with the solution for the integral term given      four or more quadrupoles in one dimension) can produce
by Eq. (A14), one can expand the various cosine terms               locally confined optics variations in between their segment.
which contain μk contributions by using the trigonometric           Since the orbit response is a specific combination of the
identity cosðx  yÞ ¼ cosðxÞ cosðyÞ ∓ sinðxÞ sinðyÞ. For the        lattice optics and it depends only on the optics at the
Jacobian elements corresponding to cases μb ; μs < μk               BPM and steerer locations as well as the tune if there exist
[labeled (A)] or μk < μb ; μs [labeled (C)], both the               such segments of quadrupoles not interleaved with BPMs
cosine terms and the integral term expand into sinð2μk Þ            nor steerers, the optics within such segments cannot be
and cosð2μk Þ terms. For the third case μb;s < μk < μs;b            resolved by observing the ORM. This can be seen in Fig. 3

                                                              032803-5
D. VILSMEIER, R. SINGH, and M. BAI                                       PHYS. REV. ACCEL. BEAMS 26, 032803 (2023)

                                                                   results, this is similar to B,Qn+,S. This pattern describes
                                                                   the placement for one section and is repeated on a section-to-
                                                                   section basis. We emphasize that this only describes in what
                                                                   order BPM, steerer, and quadrupoles are placed but it does
                                                                   not restrict the specific locations in terms of phase advance
                                                                   within each section. In fact, these specific locations may be
                                                                   different from section to section. For both dimensions,
                                                                   horizontal and vertical, we write Sh,Sv,Qn+,Bh,Bv,
                                                                   where h refers to horizontal and v refers to vertical. In terms
                                                                   of the results, this is similar to any other pattern that swaps
                                                                   any steerer with any BPM. This is because the Jacobian only
FIG. 3. K 1 L residuals when running Levenberg-Marquardt           depends on jμb − μs j and it separates horizontal from vertical
optimization for the nominal optics, starting from 1% random       contributions.
quadrupole errors and gradually leaving out BPMs and steerers         We show that the following placements exhibit a global
from consecutive sections in order to cause a pure degeneracy of   degeneracy: S,Q3+,B and Sh,Sv,Q5+,Bh,Bv. It is
the inverse problem. Each tick marker on the horizontal axis       worth noting that Sh,Sv,Q5,Bh,Bv causes a rank
indicates a quadrupole (F, D, T quadrupole per section).           deficiency of degree 1 in the Jacobian while Sh,Sv,
                                                                   Q6,Bh,Bv causes a degree 2 rank deficiency. For Sh,
which shows simulated inverse modeling results for the             Sv,Q7+,Bh,Bv, intrasection degeneracy will appear and
SIS18 lattice for all 36 quadrupoles, without any simulated        the rank of the Jacobian is the same as for Sh,Sv,Q6,
measurement uncertainty, while leaving out the BPMs and            Bh,Bv. The argument for this is similar to the one for S,
steerers from an increasing number of consecutive sections.        Q4+,B above, since exactly three column vectors are
As can be seen, for the cases where none of the sections or        needed for each dimension in order to generate the
only the first section is skipped, the quadrupole strengths        Jacobian columns for a group of consecutive quadrupoles
can be reliably recovered down to the numerical precision          in that dimension. In the Appendix, we proof the rank
of the estimator. When three or four consecutive sections          deficiency for the S,Q3+,B (Appendix C) and Sh,Sv,
are skipped, the estimates clearly become ambiguous which          Q6+,Bh,Bv (Appendix D) placements. The origin of the
is reflected by the large increase in their standard deviation.    rank deficiency for the Sh,Sv,Q5,Bh,Bv pattern is not
This is because each section contains three distinct quadru-
poles and, hence, when skipping three or more sections, the        TABLE I. This table presents an overview of the Jacobian
corresponding segment contains more than seven quadru-             properties in terms of rank deficiency for the various BPM/steerer
poles required to exhibit a degeneracy. For the case where         placements around groups of consecutive quadrupoles.
two sections are skipped, i.e., six quadrupoles, there is a
slight increase in standard deviation, similar to the amount                                                Jacobian
that is visible for the neighboring sections in the skip-3                                  No. of rows No. of columns         Rank
and skip-4 cases. This is because when the degenerate              S,Q2,B                          2              2N           2N
                                                                                                 N
segment is extended with its neighboring sections, the             S,Q3,B                        N2               3N          3N − 1
variations induced by those quadrupoles at the boundaries          S,Q4+,B                       N2              4þ N         3N − 1
of the segment are on the level of the numerical precision
of the estimator and, hence, will not be distinguished.            Sh,Sv,Q4,Bh,Bv               2N 2              4N           4N
Nevertheless, it should be noted that the order of magnitude       Sh,Sv,Q5,Bh,Bv               2N 2              5N          5N − 1
is much smaller. The saw-tooth pattern that can be observed        Sh,Sv,Q6,Bh,Bv               2N 2              6N          6N − 2
between D and T quadrupoles will be explained as                   Sh,Sv,Q7+,Bh,Bv              2N 2             7þ N         6N − 2
quasidegeneracy below.                                             N denotes the number of sections in the lattice (N ≥ 3 is
                                                                   assumed). It should be emphasized that the only deciding factor
                   1. Global degeneracy                            is the placement pattern, i.e., how many quadrupoles form a
                                                                   consecutive group, not where exactly these quadrupoles or the
   Besides the intrasection degeneracy discussed above,            BPMs/steerers are located in each of the sections. The specific
which is caused by isolated groups of consecutive quadru-          locations may vary from section to section and as long as the
poles, there can be another, global degeneracy whose               overall placement pattern is satisfied, the rank will be the same.
existence also depends on the BPM/steerer placement. In            For verification with simulations, the Jacobians were obtained
                                                                   from simulations using the MPMATH [18] library to avoid
the following, we use the notation S,Qn+,B which means
                                                                   numerical issues (dps set to 100). The rank is then
that we are considering one dimension (horizontal or               computed as the number of singular values that are larger
vertical) and the placement of lattice elements within a           than or equal to ϵN 2 smax where smax is the largest singular value
section is the following: steerer, followed by n quadrupoles       and ϵ ¼ 2−52 is the machine epsilon for double precision
(n+ means n or more), followed by a BPM. In terms of the           floating point numbers.


                                                             032803-6
INVERSE MODELING OF CIRCULAR LATTICES …                                       PHYS. REV. ACCEL. BEAMS 26, 032803 (2023)

obvious and we report this without proof, based on our                 minimization, JT J is used as an approximation of the
simulation results. Table I gives an overview of the various           Hessian H and, thus, a lower bound for the estimated
Jacobians’ ranks obtained via simulations, in agreement                parameter variance is given by σ 2 H−1. This is, of course, in
with the analytical derivations.                                       agreement since at the minimum of the cost function, the
   Appendix B includes a similar derivation for beamlines,             gradient is assumed to vanish, so the flatness of the cost
i.e., noncircular lattices.                                            function depends on how quickly that zero gradient
                                                                       changes in the neighborhood of the estimate which is
                     C. Quasidegeneracy                                indicated by the Hessian matrix.
                                                                          Figure 4 shows the JT J matrices emerging from hori-
   Even though groups of, for example, two consecutive
                                                                       zontal and vertical ORMs, together with their eigenvalue
quadrupoles do not exhibit a pure degeneracy, they can
                                                                       spectra. There are a few things to be noted. First of all, for
exhibit a quasidegeneracy which means that their estimated
                                                                       the vertical J T J plot, it can be seen that it indicates higher
strengths are much more susceptible to measurement
                                                                       variance for the D-T quadrupole pairs than for the F-D or
uncertainty than the ones of other quadrupoles. This type              F-T pairs. This is because of the scaling of the Jacobian
of quasidegeneracy is explained in the following section.              with the beta function which, in vertical, is larger at D and T
   The covariance of parameter estimates under linear least            quadrupoles than at F quadrupoles (see Fig. 1). Second, it
squares is given by σ 2 ðJT JÞ−1 where σ 2 is the variance of          can be observed that in both dimensions, there is one
observables and J is the Jacobian (if the various BPMs have            eigenvalue that is much smaller than others. Small eigen-
different measurement uncertainties, it is ðJT ΣJÞ−1 with Σ            values of JT J correspond to large eigenvalues of ðJ T JÞ−1 ,
being the covariance matrix of observables). This is closely           i.e., of the covariance estimate for model parameters.
related to the matrix JT J. The eigenvectors of a matrix and           However, for the horizontal JT J matrix, the smallest
its inverse are similar and the eigenvalues are reciprocal, so         eigenvalue in this plot is only nonzero due to limited
studying the matrix J T J reveals important information                numerical precision, since in the horizontal dimension,
about the error propagation. Also, in Gauss-Newton                     the lattice features a S,Q3,B steerer/BPM placement




FIG. 4. Top row: Horizontal dimension, bottom row: vertical dimension. Left column: The 36 × 36 matrix J T J. The axes numbering
indicates the 12 sections of SIS18 in hexadecimal notation and there are three rows/columns per section, corresponding to the F, D, and T
quadrupoles (in that order) of each section. Right column: The eigenvalues of corresponding J T J matrices. The color bars and
eigenvalue magnitude indicate the magnitude of J T J in units of m4 =rad2 . The values of the color bar correspond to those of the
eigenvalue plots shown on the vertical axes. For the horizontal dimension, the smallest eigenvalue λ35 is nonzero only due to limited
floating point precision. When inspecting the 12 smallest horizontal eigenvalues, it can be observed that the λ24 and λ25 eigenvalues have
a slightly greater magnitude than the remaining nine eigenvalues (neglecting λ35 ). These two eigenvalues correspond to sections 4 and 6
where the horizontal steerer is shifted by a few meters compared to the other sections.


                                                               032803-7
D. VILSMEIER, R. SINGH, and M. BAI                                            PHYS. REV. ACCEL. BEAMS 26, 032803 (2023)

                                                                       where the vertical lattice features a B,Q2,S BPM/steerer
                                                                       placement.
                                                                          Figure 5 shows the two eigenvectors, in horizontal and
                                                                       vertical dimensions, that correspond to the smallest eigen-
                                                                       value of the corresponding JT J matrix. Since the eigenvec-
                                                                       tors of a matrix and its inverse are similar, these indicate the
                                                                       direction of (quasi)degeneracy in both dimensions sepa-
                                                                       rately. It can be observed that this is a global degeneracy in
                                                                       both cases since all quadrupoles participate; hence, there is
                                                                       only one eigenvalue that is significantly smaller than all
                                                                       others. This is due to the symmetry of the lattice with respect
FIG. 5. Eigenvectors that correspond to the smallest eigenvalue        to the BPM/steerer placement pattern. In horizontal, for the
of the J T J matrices in horizontal (top) and vertical (bottom)        two sections 4 and 6 where the ORM’s circulant structure is
dimension. Each tick marker on the horizontal axis indicates a         broken, it can be observed that a corresponding change in the
quadrupole (F, D, and T quadrupole per section).                       quadrupoles’ degeneracy pattern reflects this. In vertical, it
                                                                       can be observed that the quasidegeneracy is driven by the
which causes a pure degeneracy (see Sec. III B 1). A zero              (noninterleaved) D-T quadrupole pairs.
eigenvalue for JT J implies a pure degeneracy since the                   Figure 6 shows the scaling of the covariance estimate for
system JT JΔp ¼ JT r (Δp parameter update, r residuals) is             model parameters, i.e., ðJT JÞ−1 ; for horizontal, since it is
underdetermined. That is, the null space of JT J is nonzero            rank deficient, ðJT J þ αIÞ−1 is plotted (with α ¼ 1 × 10−8 ,
and, hence, there exists a parameter update Δp that will               i.e., Tikhonov regularized, which is also used by, e.g., the
leave the rhs of the equation unchanged at zero. In gen-               Levenberg-Marquardt optimizer, though it uses a flexible
eral, a small eigenvalue for JT J implies a direction of               regularization parameter α). Clearly, the global nature of
quasidegeneracy which is given by the corresponding                    the degeneracy is reflected in the eigenvectors (Fig. 5).
eigenvector. It means that the parameter update emerging               From Fig. 4, it can be observed that pairwise cancelation is
from JT JΔp ¼ JT r will be susceptible to measurement                  mostly confined to nearby sections and decreases when
uncertainty in the direction of the corresponding eigen-               moving further away in terms of the phase advance.
vector. This is what is observed for the vertical Jacobian             However, the final covariance of quadrupole estimates is




FIG. 6. ðJ T J þ αIÞ−1 for the horizontal (α ¼ 1 × 10−8 ; left), vertical (α ¼ 0; middle), and combined (α ¼ 0; right) dimensions. The
axes numbering indicates the 12 sections of SIS18 in hexadecimal notation and there are three rows/columns per section, corresponding
to the F, D, and T quadrupoles (in that order) of each section. The unit of the color bars, indicating the magnitude of the matrices, is
rad2 =m4 . The quasidegeneracy pattern looks very symmetric in vertical dimension because the BPM/steerer placement is fully
symmetric from section to section. This, however, is not a requirement as shown by the horizontal data. The degeneracy pattern reflects
the differently placed steerers in sections 4 and 6. In fact, no symmetry whatsoever in terms of the exact phase advances of BPMs or
steerers is required for a degeneracy pattern to occur; only the placement pattern in terms of upstream or downstream of quadrupoles is
deciding. For the combined dimensions, it can be observed that the resulting pattern is not fully symmetric but features local correlations
slightly more than ones with other sections. This is because the magnitude of the smallest eigenvalue for the combined dimensions is
closer to the magnitude of other eigenvalues and, thus, it does not dominate the pattern alone.


                                                                032803-8
INVERSE MODELING OF CIRCULAR LATTICES …                                      PHYS. REV. ACCEL. BEAMS 26, 032803 (2023)




FIG. 7. Globally symmetric eigenmodes of J T J in vertical dimension which arise due to the fact that the vertical lattice is symmetric
from section to section. Each tick marker on the horizontal axis of the eigenvector plots indicates a quadrupole (F, D, and T quadrupole
per section).

dominated by a strong global component that is symmetric              are guaranteed to be greater than or equal to zero. This is
for the vertical ORM.                                                 observed for the vertical JT J matrix and it happens that one of
   For the vertical ORM, the corresponding J T J matrix is a          the globally symmetric eigenmodes is associated with the
block circulant matrix by the argument of section-to-                 smallest eigenvalue λ35 . Figure 7 shows the three globally
section symmetry of the vertical lattice. The eigenvectors            symmetric eigenmodes corresponding to the ρ0 ¼ 1
of a block circulant matrix B ¼ bcircðb0 ; b1 ; …; bn−1 Þ ∈           eigenvalues.
BCn;k (where n is the number of blocks and k the size                    Because for the horizontal lattice, the circulant structure
of a k × k block; n ¼ 12, k ¼ 3 in our case) are derived in           of the ORM and thus of J T J is broken in the two sections 4
[19]. They are given by                                               and 6, it cannot have a globally symmetric eigenmode, i.e.,
                        2 v 3                                         a mode that repeats on a section-to-section basis. However,
                                                                      as becomes apparent from the eigenvector Fig. 5, the global
                           6 ρm v 7                                   mode still affects all sections at once and reflects the
                           6      7
                           6 2 7                                      breaking of symmetry in sections 4 and 6.
                           6 ρm v 7                           ð7Þ
                           6      7
                           6 . 7
                           4 .. 5                                                               D. Example
                             ρn−1                                        In the absence of BPM errors, inverse modeling with an
                              m v
                                                                      optimizer such as Levenberg-Marquardt will always con-
where v is a nonzero column vector of length k, which is              verge to the ground-truth solution (within the boundaries of
given below, and ρm is one of the n complex roots of unity:           numerical precision), given that there is no additional model
ρm ¼ expð2πi mnÞ. For each ρm , there are k distinct vectors v        bias present and the initial guess is not too far from the ground
given by the following eigenvector equation [19]:                     truth (so that the optimizer will not cross any instabilities, for
                                                                      example).
        ðb0 þ ρb1 þ ρ2 b2 þ    þ ρn−1 bn−1 Þv ¼ λv         ð8Þ        Figure 8 shows the covariance of the various solutions
                                                                      obtained with the Levenberg-Marquardt optimizer when no
where λ is the corresponding eigenvalue.                              quadrupole errors are applied to the lattice and only BPM
   Since the first of the n roots of unity is ρ0 ¼ 1, from            errors are present in the ORM simulation. That is, each of
Eq. (7), it becomes apparent that every block circulant               the inverse modeling instances is given a distinct noisy
matrix B ∈ BCn;k has exactly k distinct globally symmetric            ORM emerging from the same orbit response uncertainty of
eigenmodes that repeat on a block-to-block basis. This is             7 μm mrad−1 . The initial guess is the ground-truth solution,
the case for the vertical JT J matrix.                                i.e., no quadrupole errors, but from the perspective of the
   Because J T J is real and symmetric, its eigenvalues are           optimizer, this is not the minimum of the cost function due
guaranteed to be real, too. Furthermore, since J T J is a             to the noise in the ORM; hence, it will converge to a
Gram matrix, it is positive semidefinite and its eigenvalues          different solution, the K 1 L residuals. The structure of

                                                              032803-9
D. VILSMEIER, R. SINGH, and M. BAI                                             PHYS. REV. ACCEL. BEAMS 26, 032803 (2023)




                                                                         FIG. 9. Eigenvalues of J T J for different BPM placements.
                                                                         nominal refers to the original lattice, and h-shifted refers to the
                                                                         lattice where the horizontal BPM has been shifted from its
                                                                         original position (downstream of the T quadrupole) to in between
                                                                         the D and T quadrupole. v-shifted means the same for the vertical
FIG. 8. Covariance of K 1 L residuals obtained with the
                                                                         BPM and h-v-shifted refers to both BPMs being shifted between
Levenberg-Marquardt optimizer for 7 μm mrad−1 orbit response             the D and T quadrupole. The different placement strategies vary
uncertainty when including both horizontal and vertical ORM. The         in their smallest eigenvalue which is the one that drives the
axes numbering indicates the 12 sections of SIS18 in hexadecimal         propagation of uncertainty.
notation and there are three rows/columns per section, correspond-
ing to the F, D, and T quadrupoles (in that order) of each section. No
quadrupole errors were applied to the lattice and optimization                         1. Placement of BPMs/steerers
started at the nominal quadrupole strengths. Thus, the K 1 L re-
siduals emerge purely as a result of the simulated ORM uncertainty.
                                                                            At a stage where this is still possible, the careful planning
All 36 quadrupoles have been included in the optimization.               of BPM/steerer locations can help to avoid or mitigate
                                                                         quasidegeneracy. We compare the following three scenarios
                                                                         with the results for the nominal lattice: moving either the
                                                                         horizontal or vertical BPM or both BPMs between the D
these solutions is determined by the underlying simulation               and T quadrupole. Figure 9 shows the JT J eigenvalue
model including the lattice optics. It can be seen that the              spectra for these three cases as well as for the nominal case.
quasidegeneracy is mainly driven by the D-T quadrupole                   It can be observed that the different placements of BPMs
pairs where much larger excursions in K 1 L residuals                    have different effects on the amount of quasidegeneracy.
happen. This is in agreement with Fig. 6 which shows the                 Specifically, the versions where the vertical BPMs are
predicted uncertainty from the Jacobian. For 7 μm mrad−1                 shifted between the D and T quadrupole yield significantly
orbit response uncertainty, the expected covariance of D                 smaller uncertainty in the estimated parameters while the
and T quadrupole strengths is approximately ð7 × 10−3 Þ2 ×               version with only horizontal BPMs shifted has a negligible
0.002 m−2 ≈ 1 × 10−7 m−2 . This is the amount that can                   effect. Thus, it is important to explore the different options
be observed from the simulations with the Levenberg-                     for BPM placement in order to allow for more precise
Marquardt optimizer in Fig. 8. Also, the observed covari-                inverse modeling results for future accelerators.
ance pattern matches the one from Fig. 6.
                                                                         IV. FITTING OF THE ORBIT RESPONSE MATRIX
            E. Counteracting quasidegeneracy                               The Levenberg-Marquardt optimizer uses the Jacobian at
   At different stages, different options for counteracting              every iteration. Typically, this Jacobian is computed
quasidegeneracy are feasible. During the design phase of                 numerically via finite-difference approximation, with an
the accelerator, the placement of steerers and BPMs can be               appropriate step size Δ for each parameter. In the following,
investigated in order to find a placement that reduces the               we use the analytically derived Jacobian [see Eq. (6)],
amount of quasidegeneracy compared to other placement                    which is obtained from Twiss data, for the optimization
candidates. For the SIS18 lattice, this would be achieved                procedure.1 While there is a mismatch between the
by positioning the BPMs between the D and T quadrupoles.
                                                                           1
At the stage of data analysis, the choice of optimizer allows                We note that the analytical formula Eq. (6) has been derived
for different strategies to counteract the quasidegeneracy.              under the assumption of uncoupled optics. In the presence of
                                                                         coupling, the analytical Jacobian needs to be rederived with the
Examples include introducing a cutoff during singular                    coupling terms included. Once obtained, the analytical Jacobian
value decomposition (SVD) or adding additional con-                      approach can then be applied to the inverse modeling of coupled
straints to the cost function.                                           optics.


                                                                  032803-10
INVERSE MODELING OF CIRCULAR LATTICES …                                       PHYS. REV. ACCEL. BEAMS 26, 032803 (2023)




FIG. 10. Comparison of simulation results for various cases. (A) and (N) denote, respectively, the usage of analytical or numerical
Jacobian. Q and G denote, respectively, the percentage level of random quadrupole and gain errors, uniformly sampled within these
bounds. All simulations used the Levenberg-Marquardt optimizer except the ones with the suffix (fb) which used a purely feedback-
like approach using only the analytical or numerical Jacobian obtained for the error-free model optics setting. The feedbacklike approach
converged for 67% of the simulated Q=3%, G=10% instances for both, the analytical and numerical Jacobian method. For larger
quadrupole or gain errors the rate of convergence decreases further and, hence, these results are not reported. However, simulating
quadrupole errors below 2% (not shown) results in more than 98% convergence rate for the feedbacklike approach. The convergence rate
does not depend on the simulated ORM uncertainty. All other approaches converge reliably also for the larger error levels shown in the
plot. The simulations have been performed for five different ORM uncertainties which are plotted on the horizontal axis: 0.1, 0.32, 1.0,
3.2, and 10.0 μm mrad−1 . For each uncertainty level, the eight different cases are shifted horizontally for better visibility (their order
from left to right matches the order in the legend from top to bottom); however, each case used the same ORM uncertainty for
simulations (the leftmost one). Each uncertainty level contains 100 random simulations per case.


numerical (real) and the analytical Jacobian, if this mis-             of model parameters. For an iterative scheme to
match is manageable then the fitting will still converge.              converge, the eigenvalues of the sequence of matrix
This has similarities to how closed orbit feedback (COFB)              multiplications
correction with model mismatch works [20]. In the context
of COFB, the system is assumed to be linear and there
exists a true response matrix R and a model response matrix                              ð1 − RRþ              þ
                                                                                                Θ Þk−1 …ð1 − RRΘ Þ0                    ð9Þ
RΘ . In an iterative scheme, the COFB converges if all
eigenvalues λi of 1 − RRþ  Θ fulfill −1 < λi ≤ 1 (where the            must tend to zero as k → ∞ (where k denotes the iteration
superscript þ denotes the pseudo-inverse). If R and RΘ are             count; except the m − n excess eigenvalues for rectangular
square matrices, the relationship has to be a strict inequality        R; RΘ remain at 1). This is provided if the eigenvalues of
to achieve convergence, i.e., −1 < λi < 1. Otherwise, if R             the individual matrices ð1 − RRþ   Θ Þi , for guess xi during the
and RΘ are m × n matrices with m > n, then 1 − RRþ      Θ must         ith iteration, fulfill −1 < λi ≤ 1, i.e., if the model mismatch
have the largest eigenvalue 1 with multiplicity m − n, and             is manageable for each relevant optics setting during the
all other eigenvalues must fulfill −1 < λi < 1. In the                 fitting. If the model errors are small, it might even suffice to
context of linear optics from closed orbits, the matrices              use a single Jacobian RΘ for the entire fitting procedure;
R and RΘ denote, respectively, the true and analytical                 that is, the same Jacobian can be reused during each
Jacobian. Also, the system is not entirely linear, so the              iteration.2
lattice model reacts differently to a parameter update than
the linear transformation given by R. However, if the                    2
                                                                           We note that the Matlab LOCO program [6], which is part of the
magnitude of updates is constrained, a locally linear                  Matlab Middle Layer (MML) for Accelerator Control, allows
version can be assumed at every iteration. This implies a              the user to choose whether the Jacobian should be updated during
varying true matrix R ≡ RðxÞ where x is the current guess              the fitting procedure or not.


                                                               032803-11
D. VILSMEIER, R. SINGH, and M. BAI                                         PHYS. REV. ACCEL. BEAMS 26, 032803 (2023)

   The analytical Jacobian is computed via Eq. (6) from                 Since linear coupling at SIS18 is generally corrected
Twiss data which is obtained from the accelerator model              well, the analytical expression for the Jacobian can be used
evaluated at the current parameter guess. Due to the sign            for fitting. The results are compared to those obtained by
convention for quadrupoles, for the vertical dimension, the          using the numerical Jacobian based on finite-difference
Jacobian needs to be multiplied by −1.                               approximation. The quadrupole errors are estimated with
   Using the analytical Jacobian from Twiss data is more             the Levenberg-Marquardt optimizer. Different approaches
efficient than computing the numerical Jacobian since                for mitigating the quasidegeneracy are tested in conjunction
Twiss data are computed only once for the entire                     with the analytical Jacobian approach.
Jacobian while the numerical approach computes one
ORM per quadrupole, that is, one closed orbit per steerer
                                                                                           A. Measured data
per quadrupole. Thus, the speed-up factor in terms of the
scaling with the number of relevant lattice elements is                 The ORM measurements were performed with five set-
N steerers × N quadrupoles . For the BPM and steerer gain parts of   tings per steerer, −1.0, −0.5, 0.0, 0.5, and 1.0 mrad, during a
the ORM, the analytical equation for the orbit response              long flattop of 11 s. Position data from one of the horizontal
Eq. (1) is similarly used with Twiss data in order to generate       BPMs is shown in Fig. 11. The first 2 s are skipped because
the corresponding columns of the analytical Jacobian.                the horizontal orbit still drifted during that time window; this
   Various tests with simulation data have been performed.           is likely because of the bending magnets taking a long time
The tests include random quadrupole and gain errors as               to attain their nominal strength. The long flattop duration
well as different levels of simulated orbit response uncer-          allowed for long data integration windows of 950 ms for each
tainty. The Levenberg-Marquardt algorithm has been used              steerer setting in order to reduce the measurement uncertainty.
for the fitting. The results are shown in Fig. 10. It can be         Also, sufficient time, 256 ms, was allocated for transitioning
observed that the results obtained with the analytical               between two steerer settings plus an additional 500 ms to
Jacobian match closely with those obtained with the                  allow the steerers to attain the new values. For each machine
numerical Jacobian. For the simulation case which limits             cycle, the response rc is computed from the least squares fit of
quadrupole errors by 3% and gain errors by 10%, the                  the five corresponding steerer settings. The final response r is
feedbacklike approach using only the analytical Jacobian             computed as the average over five subsequent cycles, each
obtained for the nominal optics converges in 67% of the              inversely weighted with its squared standard error σ c from the
instances and it reaches unstable lattice configurations for         least squares fit of the respective response rc :
the remaining instances. This is due to the discrepancy of
the real Jacobian with respect to the employed Jacobian                                        1 X rc
                                                                                            r¼P 1       2
                                                                                                          :                        ð10Þ
                                                                                               c σ2 c σ c
obtained from nominal optics being too large to allow
                                                                                                       c
convergence according to Eq. (9). The convergence rate is,
however, independent of the simulated ORM uncertainty.
For simulated quadrupole errors below 2%, the feedback-
like approach converges in more than 98% of instances.
Thus, this approach can be used to correct a lattice that
exhibits only small quadrupole drifts over time. For cases
with larger deviations, it is sufficient to recompute the
Jacobian during each iteration of the fitting procedure.
When the analytical Jacobian is recomputed this way, it
converges and yields good results also for larger simulated
model errors as shown in Fig. 10.

                     V. EXPERIMENT                                   FIG. 11. Position data from the horizontal BPM in section 1
                                                                     during measurement of the horizontal steerer in section 5. The
  The following experimental data have been collected to             steerer setting is overlaid as the dashed curve (the curve is
support the findings. ORM and tune measurements have                 inverted for better visibility). The two vertical axes are not
been conducted for two different optics at SIS18: nominal            aligned, i.e., there is no meaning in the vertical position of the
extraction optics and a modified version of the optics by            steerer relative to the position data. The red shaded areas indicate
adjusting one of the F quadrupole families (GS01QS1F                 the time windows available for orbit computation. The light
                                                                     shaded area (500 ms) has been excluded because the orbit was
family) by ΔK 1 L ¼ −1.2 × 10−3 m−1 (this quadrupole                 still slightly drifting during those time windows. The solid shaded
family includes the F quadrupoles from the odd-numbered              area (950 ms) is used for orbit computation. The white area
sections). Due to the very limited experimental time                 between two shaded areas is the allocated transition time for the
available, beta beating could not be measured, unfortu-              steerer magnets which is 256 ms. An additional 2 s are skipped at
nately. Nevertheless, the tune measurements serve as a               the beginning of the flattop because the horizontal orbit was still
verification for the derived quadrupole errors.                      drifting during that time window.


                                                              032803-12
INVERSE MODELING OF CIRCULAR LATTICES …                                  PHYS. REV. ACCEL. BEAMS 26, 032803 (2023)

                                                                      For each of the methods, we present the difference in
                                                                   estimates between the two optics for the F quadrupoles; that
                                                                   is, the estimates obtained for modified optics are subtracted
                                                                   by the estimates obtained for nominal optics. Both esti-
                                                                   mates are obtained by starting the fitting procedure from the
                                                                   nominal optics model. Ideally, this difference of estimates
                                                                   should be a zigzag pattern between −1.2 × 10−3 and 0 m−1
                                                                   since the GS01QS1F family contains every second F
                                                                   quadrupole (i.e., the ones from odd section numbers).

                                                                                          1. SVD cutoff
                                                                      This is performed as a two-stage process. The first
                                                                   stage uses Levenberg-Marquardt to find a (quasidegener-
                                                                   ate) solution for all the involved parameters: quadrupole
                                                                   errors and gain errors. The second stage freezes the thus
                                                                   found gain errors and restarts fitting of quadrupole errors.
                                                                   During each update step, the system JT JΔp ¼ J T r (where
                                                                   Δp is the parameter update and r is the residual vector) is
                                                                   solved by computing ðJT JÞ−1 via SVD and truncating a
                                                                   predefined number of smallest singular values to zero. If the
                                                                   SVD spectrum shows a clear drop in the magnitude of
FIG. 12. Measured tunes for nominal extraction optics (top) and    singular values then cutting the small singular values will
modified optics (bottom). The model tunes for nominal extraction   be very efficient. However, for a more flat spectrum, the
optics are 0.29 in both dimensions. The measurement for            number of singular values to cut is not obvious and also
modified optics was performed on a reduced timescale of 6 s        the resulting estimate might suffer from the truncation. This
to limit the amount of position data generated.                    strongly depends on the use case and the investigated
                                                                   lattice. The optimal cutoff value can be found from
A measurement uncertainty of 5 μm mrad−1 has been                  simulations, where random orbit uncertainties are cast on
reached for the orbit response, with minor variations among        the nominal ORM and then inverse modeling with different
the different BPMs. For the measurement of modified optics,        cutoff values is performed. The one that yields the smallest
the horizontal BPM in section 8 malfunctioned and, thus, had       error in terms of the quadrupole error estimates is then
to be removed from the analysis.                                   chosen. For our use case, we found that the best results are
   Tune measurements have been obtained by excitation via          obtained when the number of cut values is set to 11.
turn-by-turn stripline exciter and position monitoring. The
measured tunes are shown in Fig. 12. The following values                               2. ΔK1 L weights
have been measured.
                                                                       This approach adds weights to the Jacobian as described
   1. Nominal extraction optics:
                                                                   in [11]. The purpose of the weights is to limit the amount of
      (i) qh ¼ 0.3099  0.0014
                                                                   change in the ΔK 1 L parameters during each iteration of the
      (ii) qv ¼ 0.2820  0.0011
                                                                   fitting process. We determined the pattern of weights w at
   2. Modified extraction optics:
                                                                   every iteration by
      (i) qh ¼ 0.2914  0.0008
      (ii) qv ¼ 0.2871  0.0007                                                                X
                                                                                               N
                                                                                                 1
                                                                                          w¼            v;                  ð11Þ
                                                                                                      λi i
             B. Mitigation of quasidegeneracy                                                   i¼1

    To obtain meaningful results that can be compared, it is       where λi and vi are, respectively, the ith eigenvalue and
important to mitigate the quasidegeneracy which is mainly
                                                                   eigenvector of the ĴT Ĵ matrix originating from the Jacobian
driven by the D-T quadrupole pairs. We compare the
methods SVD cutoff, adding ΔK 1 L constraints to the               Ĵ that represents only the ΔK 1 L parameters and which is
Jacobian as well as leaving out T quadrupoles from the             evaluated at zero gain errors. Then wk is the weight for
fitting. The removal of T quadrupoles is justified since they      the kth quadrupole. The magnitude of w is chosen a priori
attain small strengths during extraction optics and, thus,         by a scan over different possible values and then fixed
much smaller errors are expected for this quadrupole               for every iteration. It should be emphasized that for this
family. For comparison, we refer to the results obtained           approach, we used the nominal gain Jacobian Ĵ not only for
without any method for counteracting quasidegeneracy as            the computation of the weights but it also replaced the
the baseline method.                                               ΔK 1 L part of the actual Jacobian J which is evaluated at the

                                                            032803-13
D. VILSMEIER, R. SINGH, and M. BAI                                         PHYS. REV. ACCEL. BEAMS 26, 032803 (2023)

                                                                     in place of J does not hinder convergence as their agree-
                                                                     ment is sufficiently close.

                                                                                   3. Leaving out T quadrupoles
                                                                        Since the magnitude of T quadrupole strengths is 1 order
                                                                     of magnitude smaller than the one of other quadrupoles,
                                                                     their errors are expected to be similarly smaller. Hence,
                                                                     leaving out T quadrupoles from the fitting will alter the
                                                                     estimates of other quadrupoles (mainly D quadrupoles)
                                                                     only by a relatively small amount.

                                                                                           4. Comparison
                                                                         Figure 13 shows a comparison between the three above-
                                                                     mentioned strategies for counteracting quasidegeneracy.
                                                                     Since the quasidegeneracy is mainly driven by the D-T
                                                                     quadrupole pairs, and T quadrupoles have a 1 order of
                                                                     magnitude smaller nominal strength, leaving out the T
                                                                     quadrupoles from the fit is expected to effectively eliminate
FIG. 13. Comparison of inverse modeling results for the F            the quasidegeneracy while yielding accurate results (i.e.,
quadrupoles when using different methods for counteracting           close to the actual errors). The method of adding ΔK 1 L
quasidegeneracy. Top: using analytical Jacobian. Bottom: using
                                                                     constraints to the cost function proves similarly efficient as
numerical Jacobian. The plots show the difference in estimates
for the modified optics and the nominal optics. The two optics
                                                                     it yields very similar results. The SVD cutoff method shows
differ in the manual adjustment of odd section number F              a slight deviation, mainly because the singular value
quadrupoles by ΔK 1 L ¼ −1.2 × 10−3 m−1. All other quadru-           spectrum is rather flat, and removing too many singular
poles, including the F quadrupoles from even section numbers,        values also removes too much information from the
have not been modified. The dashed lines indicate the expected       Jacobian. The same figure also shows the results obtained
(ideal) estimates for the quadrupole errors. The label Baseline      with the numerically computed Jacobian. It can be seen that
refers to the results obtained from Levenberg-Marquardt              these results closely match the results obtained with the
fitting without any countermeasure against the quasidegeneracy.      analytical Jacobian. The SVD cutoff method shows a slight
The error bars due to ORM uncertainty are on the order of            deviation between the two methods because the singular
1 × 10−5 m−1 and, thus, are not visible in the plot.                 value spectrum of the two Jacobian versions is slightly
                                                                     different.
current gain error estimate during each iteration. This is               Table II and Fig. 14 show an overview of the measured
done because when using J, the estimated gain errors                 tunes as well as the tunes obtained from the inverse
would obfuscate the degeneracy pattern of the quadrupoles            modeling results with the different methods. It can be
at every iteration. Using Ĵ, on the other hand, allows to           observed that for all methods except SVD cutoff, the
directly access the quasidegeneracy patterns and, thus,              predicted model tunes after fitting match the measured
limits them by adding corresponding weights. Using Ĵ                tunes within the measurement uncertainty. The predicted

         TABLE II.    Resulting tunes from the various fitting methods compared to measured tunes.

                                                                     Nominal optics                  Modified optics
                                                                   qh              qv               qh              qv
         Measured                         value                   0.3099         0.2820          0.2914           0.2871
                                        uncertainty               0.0014         0.0011          0.0008           0.0007
         Analytical Jacobian              Baseline                0.3098         0.2819          0.2920           0.2876
                                        SVD cutoff                0.3129         0.2822          0.2949           0.2879
                                      ΔK 1 L weights              0.3095         0.2819          0.2918           0.2876
                                      Without T quads             0.3094         0.2819          0.2917           0.2876
         Numerical Jacobian               Baseline                0.3100         0.2824          0.2917           0.2876
                                        SVD cutoff                0.3128         0.2822          0.2948           0.2879
                                      ΔK 1 L weights              0.3095         0.2819          0.2918           0.2876
                                      Without T quads             0.3094         0.2819          0.2917           0.2876



                                                           032803-14
INVERSE MODELING OF CIRCULAR LATTICES …                                    PHYS. REV. ACCEL. BEAMS 26, 032803 (2023)

                                                                     deficient and, thus, cause the inverse problem to be ill-
                                                                     defined which outlines the theoretical limitations of the
                                                                     method. In a lattice, the consecutive placement of quadru-
                                                                     poles with neither BPM nor steerer in between can cause a
                                                                     rank deficiency in the Jacobian. When fitting either the
                                                                     horizontal or vertical ORM alone, segments of three or
                                                                     more consecutive quadrupoles cause a rank deficiency.
                                                                     When fitting the horizontal and vertical ORM together,
                                                                     segments of five or more quadrupoles cause a rank
                                                                     deficiency. A rank deficiency in the Jacobian implies that
                                                                     the corresponding quadrupole strengths cannot be deter-
                                                                     mined uniquely from the measured ORM data, even in the
                                                                     absence of BPM errors.
                                                                        We have further demonstrated that the analytical expres-
                                                                     sion for the Jacobian can be used during the fitting pro-
                                                                     cedure in place of the conventional numerical Jacobian
                                                                     which is computed via finite-difference approximation. A
                                                                     single Twiss computation is sufficient to construct the
                                                                     analytical Jacobian, which allows for substantially reduced
                                                                     computation time compared to the numerical Jacobian
FIG. 14. Difference between predicted and measured tunes for         approach. The scaling of the computation in terms of the
the various optics and inverse modeling methods. Top: ΔQh ; ΔQv      number of relevant lattice elements is improved by a factor
for nominal extraction optics. Bottom: ΔQh ; ΔQv for the modi-       of N steerers × N quadrupoles by using the analytical Jacobian
fied optics. The different methods are indicated on the horizontal
                                                                     approach. The inverse modeling process by using the
axis and are the same for each subplot. (A) and (N) denote,
respectively, the usage of analytical and numerical Jacobian. The
                                                                     analytical Jacobian approach has been tested with large-
vertical bars indicate the measurement uncertainty.                  scale simulations and also with dedicated measurements
                                                                     conducted at the heavy-ion synchrotron SIS18 at GSI.
                                                                     The fitting procedure has been tested in conjunction
horizontal tune from the SVD cutoff method has a deviation           with various methods for mitigating quasidegeneracy.
of up to ≈3σ from the measured horizontal tune. This is due          The results obtained with the analytical Jacobian agree
to the rather flat singular value spectrum. The agreement of         well with those obtained with the numerical Jacobian.
predicted with measured tunes confirms that the fitted                  In summary, we explored the dependency of quaside-
models capture the global optics of the real machine. It also        generacy on the placement of BPMs and steerers and, thus,
emphasizes the effect of quasidegeneracy since also the              provide insight into how adequate numbers and locations
baseline method reproduces the measured tunes closely                for these devices can be chosen for newly designed lattices
albeit the ΔK 1 L predictions deviate significantly as can be        in order to allow for a tractable and well-conditioned
seen in Fig. 13.                                                     inverse problem. For large-scale machines, such as LHC,
                                                                     with a large number of BPMs and steerers, the size of the
                    VI. CONCLUSIONS                                  corresponding Jacobian matrix may be too big to be fully
   We studied the dependency of quasidegeneracy on the               utilizable. A practical solution is to select a subset of all
placement of BPMs and steerers for extracting linear                 ORM elements for the fitting procedure. A profound
optics and model errors from closed orbit measurements.              understanding of the impact of resulting BPM and steerer
We found that different BPM and steerer placements can               placements on the (quasi)degeneracy can help in guiding
noticeably affect the degree of quasidegeneracy and, thus,           the selection. In addition, using the analytical Jacobian
influence the quality of the lattice information that is             during fitting can provide a more computationally effi-
extracted from the measured orbit response matrix.                   cient solution for inverse modeling by circumventing the
These findings emphasize the importance of studying the              method of computing the Jacobian via finite-difference
effect of BPM and steerer placements during the design               approximation.
phase of new accelerators.
   In order to investigate the influence of BPM and steerer
                                                                              APPENDIX A: DERIVATIVE OF
placements, we derived an analytical expression for the
                                                                           ORBIT RESPONSE WITH RESPECT TO
Jacobian matrix, relating quadrupole errors along with
                                                                                QUADRUPOLE STRENGTH
BPM and steerer gain errors to the orbit response matrix.
This analytical Jacobian is then used to show which BPM                Starting with the orbit response rbs induced by steerer s
and steerer placements cause the Jacobian to be rank                 and measured by BPM b:

                                                              032803-15
D. VILSMEIER, R. SINGH, and M. BAI                                                        PHYS. REV. ACCEL. BEAMS 26, 032803 (2023)

            pﬃﬃﬃﬃﬃﬃﬃﬃﬃ        1                                                              μi ¼ μ0;i þ Δμi
    rbs ¼       βb βs                   cosðπQ − jμb − μs jÞ :             ðA1Þ                   Z s
                   ﬄ} 2 sinðπQÞ |ﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄ{zﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄ}
            |ﬄﬄﬄ{zﬄﬄ                                                                                       1
                 A     |ﬄﬄﬄﬄﬄ
                            ﬄ {zﬄﬄﬄﬄﬄ
                                    ﬄ }         C                                            μi ¼
                                                                                                       i
                                                                                                               dτ þ μs¼s0
                            B                                                                      s0 βðτÞ
                                                                                                  Z s
                                                                                                       i          1
  The derivative dðKd1 LÞ rbs ≡ r0kbs is                                                        ¼                              dτ þ μs¼s0
                                                                                                   s0    β 0 ðτÞ þ  ΔβðτÞ
                                                                                                  Z s
                             k

                                                                                                       i    1        1
                                                                                                ¼                             dτ þ μs¼s0
                                                                                                   s0 β0 ðτÞ 1 þ
                                                                                                                      ΔβðτÞ
                r0kbs ¼ A0 BC þ AB0 C þ ABC0 :                             ðA2Þ                                       β0 ðτÞ
                                                                                                  Z s                  Z s
                                                                                                      i     1                i ΔβðτÞ
                                                                                                ≈                dτ −                 dτ þ μs¼s0 ;            ðA8Þ
   In the following, the individual derivatives A0 , B0 , and C0                                   s0 β0 ðτÞ             s0 β 0 ðτÞ
                                                                                                                                    2
are derived.
                                                                                  where the subscript 0 indicates the unperturbed optics
        1                                                                         functions, i.e., without quadrupole error, and we have
A0 ¼ pﬃﬃﬃﬃﬃﬃﬃﬃﬃ ½β0b βs þ βb β0s                                                 used the fact that Taylor series are multiplicative.
     2 βb βs
                                                                                  Considering the difference μmax − μmin , we thus obtain
      β pﬃﬃﬃﬃﬃﬃﬃﬃﬃ                 β
   ≈ − k βb βs ½Ψks þ Ψkb  ¼ −A k ½Ψks þ Ψkb ;                           ðA3Þ                          Z s                       Z s
       2                           2                                                                           max     1                 max   ΔβðτÞ
                                                                                      μmax − μmin ¼                         dτ −                       dτ;    ðA9Þ
                                                                                                          smin       β0 ðτÞ          smin      β0 ðτÞ2
where we have used the formula for the beta beating [13]:
                                                                                  where smin , smax denote the corresponding longitudinal
                                cosð2πQ − 2jμk − μs jÞ                            lattice positions. Since μmax − μmin ¼ μ0;max þ Δμmax −
              β0s ≈ −βs βk                                                 ðA4Þ   μ0;min − Δμmin ¼ ðμ0;max − μ0;min Þ þ Δðμmax − μmin Þ, we
                                          2 sinð2πQÞ
                                |ﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄ{zﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄ}          obtain
                                                 Ψks
                                                                                                                             Z s
                                                                                                                                   max   ΔβðτÞ
                                                                                                   Δðμmax − μmin Þ ¼ −                           dτ:         ðA10Þ
and similarly for β0b ; Ψkb.                                                                                                  smin       β0 ðτÞ2

                                                                                  By using the expression for the beta beating this can be
                  1 cosðπQÞ          β       1
        B0 ¼ −               πQ0 ≈ −B k           ;                        ðA5Þ   rewritten as
                  2 sinðπQÞ2          2 2 tanðπQÞ
                                                                                                                         β0;k
                                                                                  Δðμmax − μmin Þ ¼ ΔðK 1 LÞk
where we have used the formula for the tune change                                                                  2 sinð2πQ0 Þ
induced by a quadrupolar error [13]:                                                                       Z s
                                                                                                               max cosð2πQ0 − 2jμ0;k − μ0 ðτÞjÞ
                                                                                                         ×                                      dτ:
                                                                                                            smin              β0 ðτÞ
                                            βk
                                  Q0 ≈                                     ðA6Þ                                                                              ðA11Þ
                                            4π
                                                                                  Approximating          the         derivative with            ðμmax − μmin Þ0 ≈
                                                                                  Δðμmax −μmin Þ
        C0 ¼ −C tanðπQ − jμb − μs jÞ                                                ΔðK 1 LÞk
                                                                                                         d
                                                                                              and using dτ μ0 ðτÞ ¼ β01ðτÞ with integration by
              
                β    μ − μmin 0                                                   substitution, we obtain
             × k − max           ðμ − μ0min Þ ;                            ðA7Þ
                4 jμmax − μmin j max
                                                                                  μ0max − μ0min
                                                                                                       Z μ
where we have assumed cosðπQ − jμb − μs jÞ ≠ 0 (i.e.,                                        β0;k            0;max
                                                                                  ¼                                  cosð2πQ0 − 2jμ0;k − ujÞdu: ðA12Þ
rbs ≠ 0) and reordered the terms μb , μs inside cosðπQ −                                2 sinð2πQ0 Þ    μ0;min
jμb − μs jÞ such that the argument of the absolute value
is positive, i.e., jμb − μs j ¼ j maxðμb ; μs Þ − minðμb ; μs Þj                  In the following, we drop the subscript 0 for nominal
and μmax ≡ maxðμb ; μs Þ; μmin ≡ minðμb ; μs Þ. In that case                      values, as there is no further ambiguity.
 μmax −μmin
jμmax −μmin j ¼ 1 and we are only left with the derivative                          Hence, all derivatives fA; B; Cg0 can be written as
μ0max − μ0min ¼ ðμmax − μmin Þ0 . To compute this derivative,                     −fA; B; Cg β2k f fA;B;Cg , i.e., the derivative r0kbs can be written
we consider the change in local phase advance Δμi induced                         as a product of rbs , the beta function at the respective
by a small quadrupolar error ΔðK 1 LÞk [21]:                                      quadrupole, and a sum of the factors f fA;B;Cg :

                                                                            032803-16
INVERSE MODELING OF CIRCULAR LATTICES …                                      PHYS. REV. ACCEL. BEAMS 26, 032803 (2023)
                 
 drbs         βk         1        tanðπQ − jμb − μs jÞ                the integrand. Therefore, we need to divide the inte-
       ¼ −rbs                  þ
dðKLÞk        2 2 tanðπQÞ                 2                           gration domain in order to resolve it. For any quadru-
                                                                      pole k, there are three distinct cases: (A) μmin <
                           tanðπQ − jμb − μs jÞ
         þ Ψks þ Ψkb −                                                μmax < μk , (B) μmin < μk < μmax , (C) μk < μmin < μmax .
                                sinð2πQÞ                              For cases (A) and (C), the argument of the absolute
           Z maxðμ ;μ Þ                         
                  b s                                                 value assumes the same sign on the entire integration
         ×              cosð2πQ − 2jμk − ujÞdu ðA13Þ                  domain and, hence, there is no need to split the
                minðμb ;μs Þ
                                                                      integration domain. For case (B), it needs to be split
   The integral in Eq. (A13) can be solved by taking                  into ½μmin ; μk  and ½μk ; μmax .
into account the absolute value function that is part of                 The solutions are

              Z μ
                  max
                  cosð2πQ − 2jμk − ujÞdu
               μmin
                  8
                  >
                  < ðAÞ sinðμmax − μmin Þ cosð2πQ − jμk − μmax j − jμk − μmin jÞ
                  >
                 ¼ ðBÞ sinðjμk − μmin jÞ cosð2πQ − jμk − μmin jÞ þ sinðjμk − μmax jÞ cosð2πQ − jμk − μmax jÞ                    ðA14Þ
                  >
                  >
                  : ðCÞ sinðμ − μ Þ cosð2πQ − jμ − μ j − jμ − μ jÞ
                                    max     min                  k    max      k    min



Hence, the result for cases (A) and (C) is similar and a                 Compared with the Jacobian for a circular lattice, the
distinction has to be made between the two different cases            beamline Jacobian additionally has some of its elements
(A and C) for which both μmin , μmax are either upstream or           zeroed. Thus, the rank of the beamline Jacobian for a given
downstream of the quadrupole and (B) for which μmin is                BPM/steerer placement must be less than or equal to the rank
upstream and μmax is downstream of the quadrupole.                    of the corresponding circular lattice Jacobian. Our simula-
                                                                      tions show that it is rank deficient for the cases Sh,Sv,Q5+,
                                                                      Bh,Bv but has full rank for Sh,Sv,Q4,Bh,Bv.
    APPENDIX B: DERIVATIVE OF ORBIT
 RESPONSE WITH RESPECT TO QUADRUPOLE
                                                                            APPENDIX C: PROOF: S,Q3,B JACOBIAN
       STRENGTH FOR BEAMLINES
                                                                                    IS RANK DEFICIENT
  For beamlines, or more generally, nonclosed lattices, we
                                                                         The trigonometric expressions in the Jacobian [Eq. (6)]
have the following formula for the orbit response at BPM b
                                                                      can be expanded in terms of μk by using the identities
induced by steerer s [22]:
                                                                      cosðx  yÞ ¼ cosðxÞ cosðyÞ ∓ sinðxÞ sinðyÞ, sinðx  yÞ ¼
                pﬃﬃﬃﬃﬃﬃﬃﬃﬃ                                           sinðxÞ cosðyÞ  cosðxÞ sinðyÞ, sinð2xÞ ¼ 2 sinðxÞ cosðxÞ,
                    βb βs sinðμb − μs Þ; μb > μs
        rbs ¼                                         ðB1Þ            cosð2xÞ ¼ cosðxÞ2 − sinðxÞ2 , 1 ¼ cosðxÞ2 þ sinðxÞ2 . The
                 0;                       otherwise
                                                                      resulting expression can be grouped by terms containing
   The relation for Δβ                                                cosðμk Þ2 , − sinðμk Þ2 , and 2 cosðμk Þ sinðμk Þ. This allows to
                     β for nonclosed lattices to first order is
given by [23]:                                                        represent each column of the Jacobian by a set of three
                                                                      coefficient vectors, one for each of the trigonometric terms.
               Δβx                                                    These coefficient vectors contain the phase advances of
                       ¼ −βk βx sinð2μx − 2μk Þ;             ðB2Þ
             ΔðK 1 LÞk                                                BPMs/steerers and their structure only depends on whether
                                                                      the BPM/steerer placement is of type A (μmin < μmax < μk ),
where the subscript x refers to the point of measurement              type B (μmin < μk < μmax ), or type C (μk < μmin < μmax ),
and k refers to the quadrupole; μx > μk is assumed since              where μmin ≡ minðμb ; μs Þ and μmax ≡ maxðμb ; μs Þ. Since
only downstream regions are affected.                                 the quadrupole triplets of S,Q3,B are not interleaved by
   Taking the derivative of rbs with respect to ΔðK 1 LÞk one         BPMs/steerers, the structure of coefficient vectors is the
obtains the following:                                                same for each quadrupole in a triplet. In fact, these three
                       0;                                  μk < μs   coefficients vectors can be used for more than three
          drbs
rkbs ≡             ¼            sinðμb −μk Þ sinðμk −μs Þ
                                                                      consecutive quadrupoles as well since the coefficient
       dΔðK 1 LÞk       −rbs βk       sinðμb −μs Þ        ; μk > μs   vectors only need to be multiplied by the three trigono-
                                                             ðB3Þ     metric factors containing μk for a given quadrupole in order
                                                                      to generate the corresponding column of the Jacobian.
  This can be expanded into cosðμk Þ2 , sinðμk Þ2 , and               Hence, this proof applies to S,Q3+,B BPM/steerer place-
cosðμk Þ sinðμk Þ terms with their respective coefficient vectors.    ments as well. Thus, one set of three coefficient vectors is

                                                               032803-17
D. VILSMEIER, R. SINGH, and M. BAI                                            PHYS. REV. ACCEL. BEAMS 26, 032803 (2023)

                                                                        sufficient to generate the Jacobian columns for a full quadru-
                                                                        pole n-tuplet with n ≥ 3. This means that there are a total of
                                                                        3N coefficient vectors, one 3-tuple per quadrupole n-tuplet in
                                                                        each of the N sections. These column vectors form the
                                                                        column span of any S,Qn+,B Jacobian for n ≥ 3. The
                                                                        structure of these coefficient vectors, in terms of the phase
                                                                        advance types A, B, C, is shown exemplarily for N ¼ 4,
                                                                        n ¼ 3 in schematic (Fig. 15).
                                                                           We use the following set of abbreviations to simplify the
                                                                        notation:

                                                                            u ≡ μmax þ μmin
                                                                            v ≡ μmax − μmin
                                                                            T ≡ tanðπQ − jμmax − μmin jÞ ¼ tanðπQ − vÞ
                                                                                        1     T
                                                                            T̃ ≡            þ                                   ðC1Þ
                                                                                   2 tanðπQÞ 2

                                                                          Further, (1) is used to represent cosðμk Þ2 , (2) for
                                                                        − sinðμk Þ2 , and (3) for 2 cosðμk Þ sinðμk Þ.
                                                                          The specific expressions for the coefficient vectors, in
                                                                        dependence on the trigonometric factor (1, 2, 3) and type
                                                                        (A, B, C), are shown in Table III.
                                                                          The expressions in Table III can be further simplified by
                                                                        noting the following relationships:

                                                                                                           cosðπQÞ
                                                                                        cosðvÞ − T sinðvÞ ¼
                                                                                                         cosðπQ − vÞ
                                                                                                           cosðπQÞ
                                                                        cosð2πQ − vÞ þ T sinð2πQ − vÞ ¼
                                                                                                         cosðπQ − vÞ
                                                                                                           cosðπQÞ
                                                                                                    T̃ ¼
                                                                                                         cosðπQ − vÞ
FIG. 15. This schematic shows the Jacobian elements’ types A,
B,C for N ¼ 4 sections and n ¼ 3 quadrupoles forming a triplet                                                   cosðvÞ
                                                                                                         ×
in each of the sections. The quadrupoles in a triplet are labeled F,                                       2 sinðπQÞ cosðπQÞ
D,T. [i] stands for the ith BPM and <i> stands for the ith
                                                                                                                                ðC2Þ
steerer. As can be seen, the quadrupoles within a triplet all share
the same type for each BPM/steerer pair.



          TABLE III. Expressions for the coefficient vectors for the different types A, B, C. The relationship cosðxÞ þ
          cosðyÞ ¼ 2 cosðxþy
                           2 Þ cosð 2 Þ has been used to combine the cos terms originating from the Ψks and Ψkb terms. Note
                                   x−y

          that for each (A, B, C), the only difference in the (1) and (2) expressions is the sign of the trailing terms.

          (1)                         A                                     2 cosð2πQ þ uÞ½cosðvÞ − T sinðvÞ þ T̃
                                      B                          2 cosðuÞ½cosð2πQ − vÞ þ T sinð2πQ − vÞ − 2T sinð2πQÞ þ T̃
                                      C                                     2 cosð2πQ − uÞ½cosðvÞ − T sinðvÞ þ T̃
          (2)                         A                                     2 cosð2πQ þ uÞ½cosðvÞ − T sinðvÞ − T̃
                                      B                          2 cosðuÞ½cosð2πQ − vÞ þ T sinð2πQ − vÞ þ 2T sinð2πQÞ − T̃
                                      C                                     2 cosð2πQ − uÞ½cosðvÞ − T sinðvÞ − T̃
          (3)                         A                                       2 sinð2πQ þ uÞ½cosðvÞ − T sinðvÞ
                                      B                                   2 sinðuÞ½cosð2πQ − vÞ þ T sinð2πQ − vÞ
                                      C                                      −2 sinð2πQ − uÞ½cosðvÞ − T sinðvÞ


                                                                032803-18
INVERSE MODELING OF CIRCULAR LATTICES …                                          PHYS. REV. ACCEL. BEAMS 26, 032803 (2023)

TABLE IV. Simplified expressions for the coefficient vectors               can create a further simplified matrix that consists of the
                                                        cosðπQÞ
for the different types (A, B, C). The common factor 2 cosðπQ−vÞ           expressions in Table IV with these terms removed and
has been removed from the expressions in Table III.                        augmentedPby an additional row which enforces the
                                                  cosðvÞ                   condition        X∈fA;B;Cg ðρX − σ X Þ ¼ 0 which allowed the
(1)         A                cosð2πQ þ uÞ þ 4 sinðπQÞ cosðπQÞ
                                       cosðvÞ        sinð2πQÞ sinðπQ−vÞ
                                                                           removal of those terms. The new version is shown in
            B          cosðuÞ þ ½4 sinðπQÞ cosðπQÞ −       cosðπQÞ        Table V. It should be noted that this is not an equivalence
                                                  cosðvÞ
            C                cosð2πQ − uÞ þ 4 sinðπQÞ cosðπQÞ
                                                                           transformation, but the null-space of the new matrix is
                                                                           contained in the null-space of the original matrix. Hence, it
                                                  cosðvÞ
(2)         A                cosð2πQ þ uÞ − 4 sinðπQÞ cosðπQÞ              is sufficient to show that the new matrix represented by
            B                          cosðvÞ
                       cosðuÞ − ½4 sinðπQÞ cosðπQÞ −
                                                     sinð2πQÞ sinðπQ−vÞ
                                                           cosðπQÞ        Table V is rank deficient.
            C                                     cosðvÞ
                             cosð2πQ − uÞ − 4 sinðπQÞ
                                                                               We can reorder the various terms of J̃ to construct a new
                                                      cosðπQÞ
                                                                           matrix M̃ such that the columns of M̃ correspond to ρi þ σ i ,
(3)         A                          sinð2πQ þ uÞ                        τi , and ρi − σ i (in that order), where i refers to the ith column
            B                              sinðuÞ                          of the three matrices containing all type-(1,2,3) terms. This
            C                         − sinð2πQ − uÞ
                                                                           reordering preserves the dot product J̃ · v⃗ ¼ M̃ · v⃗ . Only the
                                                                           ρi − σ i terms depend on v while the other terms depend on u.
             cosðπQÞ                                                       The overall matrix thus consists of a column-wise stack of
    Thus, 2 cosðπQ−vÞ is a common factor for all expressions in            three submatrices corresponding to ρi þ σ i , τi , and ρi − σ i
Table III and removing this factor does not alter the rank of              and has the following form:
the matrix. Therefore, we obtain the simplified expressions                                          ρþσ
shown in Table IV.                                                                                    M        M τ Mρ−σ
                                                                                             M̃ ¼                                         ðC3Þ
    Let J̃ be the column-wise stack of the 3N coefficient                                             0…0 0…0 1…1
vectors emerging from the simplified expressions in
Table IV. Since all the used simplifications preserved the                 PThe additional last row enforces the condition
column span of the Jacobian (up to constant factors), the                     X∈fA;B;Cg ðρX − σ X Þ ¼ 0. While the original Jacobian J
nullspace and, thus, the rank of J̃ is similar to that of                  has shape N 2 × 3N (for N sections), the new matrix M̃ has
the original Jacobian J. Thus, it is sufficient to show that J̃            shape ðN 2 þ 1Þ × 3N. By the above derivation, it has,
is rank deficient, i.e., that there exists a vector v⃗ such that           however, the same nullspace as J. Thus, it is sufficient to
          ⃗ This matrix multiplication involves the row-wise
J̃ · v⃗ ¼ 0.                                                               show that M̃ is rank deficient. Because the rank of a matrix
summation of the various coefficient vectors that make up                  does not change under row- or column-wise multiplication
the matrix J̃. Each row contains at most the three distinct                with a nonzero constant, the common factor sinð2πQÞ
                                                                                                                          cosðπQÞ can be
types A,B,C (see                                                                                     ρ−σ
                  Pschematic 15). Thus, eachProw-wise sum                  removed from the M            matrix leaving it with only
is of the form X∈fA;B;Cg ρX · fð1Þ; Xg þ X∈fA;B;Cg σ X ·                   sinðπQ − vÞ terms.
              P
fð2Þ; Xg þ X∈fA;B;Cg τX · fð3Þ; Xg where ρX stands for                        Since the Gram matrix AT A of any m × n matrix A
the sum of entries in v⃗ corresponding to type fð1Þ; Xg in the             (m ≥ n) has the same rank as the original matrix A, it is
coefficient matrix and similarly
                           P       σ refers to type (2) and τ to           sufficient to show that the Gram matrix of M̃ is rank
type (3). If we require X∈fA;B;Cg ðρX − σ X Þ ¼ 0, then the                deficient. Since the Gram matrix is a square matrix, its
                      cosðvÞ
terms involving 4 sinðπQÞ                                                  determinant can be computed from the original matrix via
                          cosðπQÞ in Table IV vanish. Thus, we
                                                                           the Cauchy-Binet formula [24]:
                                                                                                       X
TABLE V. Further simplified expressions for the coefficient                         detðM̃T M̃Þ ¼           det ðM̃½αjnÞ2 ¼ 0 ðC4Þ
vectors
P       for the different types A,B,C. The additional requirement                                   α∈INCðm;nÞ
  X∈fA;B;Cg ðρX − σ X Þ ¼ 0 has to be satisfied.
                                                                           where n denotes the set of numbers f1; 2; …; ng and
(1)               A                          cosð2πQ þ uÞ                  INCðm; nÞ denotes the set of all strictly increasing func-
                  B                   cosðuÞ − sinð2πQÞ
                                               cosðπQÞ sinðπQ − vÞ         tions from m to n; M̃½αjn denotes the submatrix of M̃ that
                  C                          cosð2πQ − uÞ                  emerges from selecting the rows with indices given by α
(2)               A                         cosð2πQ þ uÞ                   and column indices given by n.
                                                                               Equation (C4) implies that the determinants of all
                  B                  cosðuÞ þ sinð2πQÞ
                                              cosðπQÞ sinðπQ − vÞ
                                                                           individual submatrices M̃½αjn need to be zero.
                  C                         cosð2πQ − uÞ
                                                                               To further simplify the involved expressions, we make
(3)               A                           sinð2πQ þ uÞ                 use of the identities cosðxÞ ¼ 12 ðeix þ e−ix Þ and sinðxÞ ¼
                  B                               sinðuÞ                   1          −ix
                                                                           2i ðe − e      Þ which allow to replace the various cos, sin
                                                                                ix
                  C                          − sinð2πQ − uÞ
                                                                           terms with the following expressions:

                                                                   032803-19
D. VILSMEIER, R. SINGH, and M. BAI                                           PHYS. REV. ACCEL. BEAMS 26, 032803 (2023)




FIG. 16. PARI/GP program for verifying that the determinant of every 9 × 9 submatrix of the 10 × 9 M̃ matrix for the N ¼ 3 case is
identical to zero. The simplifications from Eq. (C6) have been applied. The following abbreviations are used: fa; b; cg ≡ eiμb;f1;2;3g ,
fd; e; fg ≡ eiμs;f1;2;3g , g ≡ eiπQ . PARI/GP version 2.13.4 has been used. The program can be run by copying it into a file main.gp and
then running path/to/gp2c-run main.gp followed by typing compute().


                                        p2 q2 þ 1                     whose inverse occurs in every element across a row. Note
                 cosðμmax þ μmin Þ ¼                                  that these elementary row/column operations preserve
                                           2pq
                                                                      the rank of the matrix. This yields the further simplified
                                        p2 q2 g4 þ 1
         cosð2πQ þ μmax þ μmin Þ ¼                                    expressions given by
                                           2pqg2
                                    p2 q2 þ g4                                         cosðμmax þ μmin Þ → p2 q2 g2 þ g2
          cosð2πQ − μmax − μmin Þ ¼
                                     2pqg2                                     cosð2πQ þ μmax þ μmin Þ → p2 q2 g4 þ 1
                                        p2 q2 − 1                              cosð2πQ − μmax − μmin Þ → p2 q2 þ g4
                  sinðμmax þ μmin Þ ¼
                                         2ipq
                                                                                        sinðμmax þ μmin Þ → p2 q2 g2 − g2
                                   p2 q2 g4 − 1
         sinð2πQ þ μmax þ μmin Þ ¼                                             sinð2πQ þ μmax þ μmin Þ → p2 q2 g4 − 1
                                    2ipqg2
                                                                                sinð2πQ − μmax − μmin Þ → p2 q2 − g4
                                    p2 q2 − c4
          sinð2πQ − μmax − μmin Þ ¼                                              sinðπQ − μmax þ μmin Þ → q2 g2 − p2 :            ðC6Þ
                                     2ipqg2
                                          2 2     2
                                        q g −p                           Thus, the resulting matrix, with cos, sin terms being
           sinðπQ − μmax þ μmin Þ ¼             ;           ðC5Þ
                                          2ipqg                       replaced by Eq. (C6), contains only various polynomial
                                                                      terms as elements. With the help of a computer algebra
where p ≡ eiμmax , q ≡ eiμmin , g ≡ eiπQ for the given values         system such as PARI/GP [25], it can be shown that the
                                                                      determinants of all 9 × 9 submatrices of the simplified
of μmax , μmin in each row.
   It is sufficient to show the rank deficiency for the               10 × 9 matrix M̃ are identical to zero. From this follows
N ¼ 3, n ¼ 3 (i.e., three sections containing quadrupole              that M̃ is rank deficient, according to Eq. (C4). An example
triplets) case; the general case N > 3 follows from the               program is given by program (Fig. 16).
symmetric placement of lattice elements from one section                 It is worth noting that the proof does not make any
to another and n > 3 follows from the fact that the same set          assumptions on the values of μb;j , μs;j , and Q. Thus, the
of three coefficient vectors is sufficient to generate the            rank deficiency holds for arbitrary values of μb;j , μs;j , and Q
Jacobian columns of any quadrupole n-tuplet, i.e., M̃ is a            and does not restrict the optics nor the specific placement of
ðN 2 þ 1Þ × 3N matrix independent of n.                               BPMs or steerers in terms of their phase advance.
   The expressions in Eq. (C5) can be further simplified by
multiplying columns 1,2,3 of M̃ (containing only cos                        APPENDIX D: PROOF: Sh,Sv,Q6,Bh,Bv
terms) by 2g2, columns 4,5,6 (containing only sin terms)                       JACOBIAN IS RANK DEFICIENT
by 2ig2, and columns 7,8,9 (containing only sin terms) by               The proof for the Sh,Sv,Q6,Bh,Bv placement is
2ig. Then the first row can be multiplied by ð2igÞ−1 and              analogous to the one obtained for S,Q3,B (Appendix C).
each other row can be multiplied by their respective pq               Instead of three coefficient vectors, there are six coefficient

                                                             032803-20
INVERSE MODELING OF CIRCULAR LATTICES …                                   PHYS. REV. ACCEL. BEAMS 26, 032803 (2023)

vectors, three for each dimension. These coefficient vectors        [11] X. Huang, J. Safranek, and G. Portman, LOCO with
are orthogonal since the horizontal coefficient vectors only             constraints and improved fitting technique, ICFA Beam
have nonzero entries in the horizontal part of the Jacobian              Dyn. Newslett. 44, 60 (2007).
while the vertical coefficient vectors only have nonzero            [12] D. Vilsmeier, ACCINV (version 0.1.0.post1), 2022.
                                                                         https://pypi.org/project/accinv/.
entries in the vertical part of the Jacobian and the two parts of
                                                                    [13] S. Y. Lee, Accelerator Physics (Fourth Edition) (WSPC,
the Jacobian are entirely separate. Hence, we can construct a            Singapore, 2018).
matrix similar to M̃ in Eq. (C3) but now the matrix is a block      [14] P. Spiller et al., The FAIR Heavy Ion Synchrotron SIS100,
diagonal of shape ð2N 2 þ 2Þ × 6N where the upper-left                   J. Instrum. 15, T12013 (2020).
block is the M̃ for the horizontal dimension and the                [15] R. Bär, K. Blasche, H. Eickhoff, B. Franczak, I. Hofmann,
lower-right block is the M̃ for the vertical dimension.                  P. Moritz, A. Dolinski, and A. Dymnikov, SIS operation at
Both blocks independently induce a rank deficiency as                    high beam intensities, in Proceedings of the 6th European
                                                                         Particle Accelerator Conference, Stockholm, 1998 (IOP,
shown in Appendix C. Thus, the rank deficiency for the
                                                                         London, 1998), pp. 499–501.
Sh,Sv,Q6,Bh,Bv Jacobian is twice the one for S,Q3,B.                [16] A. Reiter and R. Singh, Comparison of beam position
                                                                         calculation methods for application in digital acquisition
                                                                         systems, Nucl. Instrum. Methods Phys. Res., Sect. A 890,
                                                                         18 (2018).
 [1] J. Safranek, Experimental determination of storage ring        [17] P. Amstutz, T. Plath, S. Ackermann, J. Bödewadt, C.
     optics using orbit response measurements, Nucl. Instrum.            Lechner, and M. Vogt, Confining continuous manipula-
     Methods Phys. Res., Sect. A 388, 27 (1997).                         tions of accelerator beam-line optics, Phys. Rev. Accel.
 [2] L. S. Nadolski, LOCO fitting challenges and results for             Beams 20, 042802 (2017).
     SOLEIL, ICFA Beam Dyn. Newslett. 44, 69 (2007).                [18] F. Johansson et al., MPMATH: A Python library for
 [3] M. Spencer, LOCO at the Australian Synchrotron, ICFA                arbitrary-precision floating-point arithmetic (version 1.2.1),
     Beam Dyn. Newslett. 44, 81 (2007).                                  February 2021, http://mpmath.org/.
 [4] R. Dowd, M. Boland, G. LeBlanc, and Y. Tan, Achieve-           [19] G. J. Tee, Eigenvectors of block circulant and alter-
     ment of ultralow emittance coupling in the Australian               nating circulant matrices, Res. Lett. Inf. Math. Sci. 8,
     Synchrotron storage ring, Phys. Rev. ST Accel. Beams 14,            123 (2005).
     012804 (2011).                                                 [20] S. H. Mirza, R. Singh, P. Forck, and B. Lorentz, Perfor-
 [5] M. Aiba, M. Böge, J. Chrin, N. Milas, T. Schilcher, and A.          mance of the closed orbit feedback systems with spatial
     Streun, Comparison of linear optics measurement and                 model mismatch, Phys. Rev. Accel. Beams 23, 072801
     correction methods at the Swiss Light Source, Phys.                 (2020).
     Rev. ST Accel. Beams 16, 012802 (2013).                        [21] O. S. Brüning, Linear imperfections, in Proceedings of
 [6] J. Safranek, G. Portmann, A. Terebilo, and C. Steier,               CAS-CERN Accelerator School:Intermediate Accelerator
     MATLAB-based LOCO, in Proceedings of the 8th Euro-                  Physics, Tuusula, Finland (CERN, Geneva, 2006).
     pean Particle Accelerator Conference, Paris, 2002              [22] V. Sajaev, Simulation of linear lattice correction of an
     (EPS-IGA and CERN, Geneva, 2002), pp. 1184–1186.                    energy-recovery linac designed for an APS upgrade, in
 [7] X. Yang, V. Smaluk, L. H. Yu, Y. Tian, and K. Ha, Fast and          Proceedings of the 24th Linear Accelerator Conference
     precise technique for magnet lattice correction via sine-           LINAC’08, Victoria, BC, Canada (JACoW, Geneva, Swit-
     wave excitation of fast correctors, Phys. Rev. Accel. Beams         zerland, 2008).
     20, 054001 (2017).                                             [23] R. Tomas Garcia, A. Garcia-Tabares Valdivieso, A. S.
 [8] I. Ziemann and V. Ziemann, Noninvasively improving the              Langner, L. Malina, and A. Franchi, Average beta-
     orbit-response matrix while continuously correcting the             beating from random errors, CERN Report No. CERN-
     orbit, Phys. Rev. Accel. Beams 24, 072804 (2021).                   ACC-NOTE, 2018
 [9] X. Huang, Linear optics and coupling correction with           [24] J. G. Broida and S. G. Williamson, A Comprehensive
     closed orbit modulation, Phys. Rev. Accel. Beams 24,                Introduction to Linear Algebra (Addison-Wesley, Reading,
     072805 (2021).                                                      MA, 1986).
[10] X. Huang, Beam diagnosis and lattice modeling of the           [25] The PARI Group, Univ. Bordeaux, PARI/GP version 2.13.4,
     Fermilab booster. Ph.D. thesis, Indiana University,                 2022, available from https://pari.math.u-bordeaux.fr/.
     Bloomington, IN, 2005.




                                                             032803-21
