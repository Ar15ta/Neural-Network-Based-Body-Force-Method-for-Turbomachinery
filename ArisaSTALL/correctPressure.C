/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2022-2023 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "ArisaSTALL.H"
#include "constrainHbyA.H"
#include "constrainPressure.H"
#include "adjustPhi.H"
#include "fvcMeshPhi.H"
#include "fvcFlux.H"
#include "fvcDdt.H"
#include "fvcGrad.H"
#include "fvcSnGrad.H"
#include "fvcReconstruct.H"
#include "fvcVolumeIntegrate.H"
#include "fvmDiv.H"
#include "fvmLaplacian.H"

// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::solvers::ArisaSTALL::correctBlockageDensity
(
    const surfaceScalarField& lambdaf
)
{
    fvScalarMatrix rhoEqn
    (
        lambda_*fvm::ddt(rho_) + fvc::div(lambdaf*phi_)
      ==
        lambda_*fvModels().source(rho_)
    );
    fvConstraints().constrain(rhoEqn);
    rhoEqn.solve();
    fvConstraints().constrain(rho_);
}


void Foam::solvers::ArisaSTALL::prePredictor()
{
    if (buoyancy.valid())
    {
        fluid::prePredictor();
    }
    else if (!mesh.schemes().steady() && !pimple.simpleRho() && pimple.firstIter())
    {
        correctBlockageDensity(fvc::interpolate(lambda_)());
    }
}


void Foam::solvers::ArisaSTALL::pressureCorrector()
{
    if (buoyancy.valid())
    {
        fluid::pressureCorrector();
        return;
    }

    while (pimple.correct())
    {
        correctPressure();
    }
    tUEqn.clear();
}


void Foam::solvers::ArisaSTALL::correctPressure()
{
    volScalarField& rho(rho_);
    volScalarField& p(p_);
    volVectorField& U(U_);
    surfaceScalarField& phi(phi_);

    const volScalarField& lambda = lambda_;
    const surfaceScalarField lambdaf(fvc::interpolate(lambda));

    const volScalarField& psi = thermo.psi();
    rho = thermo.rho();
    rho.relax();

    fvVectorMatrix& UEqn = tUEqn.ref();

    // Thermodynamic density needs to be updated by psi*d(p) after the
    // pressure solution
    const volScalarField psip0(psi*p);

    const surfaceScalarField rhof(fvc::interpolate(rho));

    const volScalarField rAU("rAU", 1.0/UEqn.A());
    const surfaceScalarField rhorAUf("rhorAUf", fvc::interpolate(rho*rAU));

    tmp<volScalarField> rAtU
    (
        pimple.consistent()
      ? volScalarField::New("rAtU", 1.0/(1.0/rAU - UEqn.H1()))
      : tmp<volScalarField>(nullptr)
    );

    tmp<surfaceScalarField> rhorAtUf
    (
        pimple.consistent()
      ? surfaceScalarField::New("rhoRAtUf", fvc::interpolate(rho*rAtU()))
      : tmp<surfaceScalarField>(nullptr)
    );

    const volScalarField& rAAtU = pimple.consistent() ? rAtU() : rAU;
    const surfaceScalarField& rhorAAtUf =
        pimple.consistent() ? rhorAtUf() : rhorAUf;
    const surfaceScalarField pressureMobility(lambdaf*rhorAAtUf);

    volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U, p));

    if (pimple.nCorrPiso() <= 1)
    {
        tUEqn.clear();
    }

    surfaceScalarField phiHbyA
    (
        "phiHbyA",
        rhof*fvc::flux(HbyA)
      + fvc::interpolate(lambda*rho*rAU)*fvc::ddtCorr(rho, U, phi, rhoUf)
    );

    MRF.makeRelative(rhof, phiHbyA);

    tmp<surfaceScalarField> phid;
    if (pimple.transonic())
    {
        const surfaceScalarField phidByPsi
        (
            constrainPhid(fvc::relative(phiHbyA, rho, U)/rhof, p)
        );
        phid = surfaceScalarField::New
        (
            "phid", lambdaf*fvc::interpolate(psi)*phidByPsi
        );
        phiHbyA -= fvc::interpolate(psi*p)*phidByPsi;
    }

    const surfaceScalarField snGradLambdaP(fvc::snGrad(lambda*p));

    if (pimple.consistent())
    {
        phiHbyA += (rhorAAtUf - rhorAUf)*snGradLambdaP*mesh.magSf();
        HbyA += (rAAtU - rAU)*fvc::grad(lambda*p);
    }

    // Keep p as the matrix unknown. Lag only the part of the discrete
    // product gradient that the implicit pressure Laplacian does not supply.
    phiHbyA -= rhorAAtUf*mesh.magSf()
        *(snGradLambdaP - lambdaf*fvc::snGrad(p));

    constrainPressure(p, rho, U, phiHbyA, pressureMobility, MRF);
    fvc::makeRelative(phiHbyA, rho, U);

    // Continuity transports the open-area flux. phi itself stays per full
    // face for ddtCorr, rhoUf and the transport equations that multiply by lambda.
    surfaceScalarField lambdaPhiHbyA(lambdaf*phiHbyA);
    const bool adjustMass =
        !pimple.transonic() && mesh.schemes().steady()
     && adjustPhi(lambdaPhiHbyA, U, p);

    fvScalarMatrix pDDtEqn
    (
        lambda*fvc::ddt(rho) + psi*correction(lambda*fvm::ddt(p))
      + fvc::div(lambdaPhiHbyA)
     ==
        lambda*fvModels().sourceProxy(rho, p)
    );

    if (pimple.transonic())
    {
        pDDtEqn += fvm::div(phid(), p);
    }

    while (pimple.correctNonOrthogonal())
    {
        fvScalarMatrix pEqn
        (
            pDDtEqn - fvm::laplacian(lambdaf*pressureMobility, p)
        );
        if (pimple.transonic())
        {
            pEqn.relax();
        }

        pEqn.setReference
        (
            pressureReference.refCell(),
            pressureReference.refValue()
        );
        fvConstraints().constrain(pEqn);
        pEqn.solve();

        if (pimple.finalNonOrthogonalIter())
        {
            phi = (lambdaPhiHbyA + pEqn.flux())/lambdaf;
        }
    }

    if (!mesh.schemes().steady())
    {
        const bool constrained = fvConstraints().constrain(p);

        thermo_.correctRho(psi*p - psip0);

        if (constrained)
        {
            rho = thermo.rho();
        }

        // Correct density to satisfy continuity after all other rho updates
        correctBlockageDensity(lambdaf);
    }

    fluidSolver::continuityErrors
    (
        (lambda*rho)(), (lambda*thermo.rho())(), (lambdaf*phi)()
    );

    p.relax();

    if (mesh.schemes().steady())
    {
        fvConstraints().constrain(p);
    }

    if (adjustMass && !thermo.incompressible())
    {
        p += (initialMass - fvc::domainIntegrate(lambda*thermo.rho()))
            /fvc::domainIntegrate(lambda*psi);
        p.correctBoundaryConditions();
    }

    U = HbyA - rAAtU*fvc::grad(lambda*p);
    U.correctBoundaryConditions();
    fvConstraints().constrain(U);
    K = 0.5*magSqr(U);

    if (mesh.schemes().steady() || pimple.simpleRho() || adjustMass)
    {
        rho = thermo.rho();
    }

    if (mesh.schemes().steady() || pimple.simpleRho())
    {
        rho.relax();
    }

    fvc::correctRhoUf(rhoUf, rho, U, phi, MRF);

    if (thermo.dpdt())
    {
        dpdt = fvc::ddt(p);
    }
}

// ************************************************************************* //
