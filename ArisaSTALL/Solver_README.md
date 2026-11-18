# ArisaSTALL - 带Benneke金属堵塞修正的可压缩欧拉求解器

## 1. 核心技术原理

### 1.1 堵塞因子 λ 的物理意义
λ 表示流体体积分数（0 < λ ≤ 1），用于模拟金属堵塞对流体流动的影响。在 Benneke 模型中，λ 直接影响动量方程和能量方程的各项，相当于减小了流体的有效流通面积。

### 1.2 数学推导

#### 1.2.1 动量方程修正
原始动量方程：
$
\frac{\partial (\rho \mathbf{U})}{\partial t} + \nabla \cdot (\rho \mathbf{U} \mathbf{U}) = -\nabla p + \rho \mathbf{g} + S
$

添加 λ 修正后：
$
\frac{\partial (\lambda \rho \mathbf{U})}{\partial t} + \nabla \cdot (\lambda \rho \mathbf{U} \mathbf{U}) = -\nabla (\lambda p) + \lambda \rho \mathbf{g} + \lambda S
$

#### 1.2.2 压力梯度项处理
根据链式法则：
$
\nabla (\lambda p) = \lambda \nabla p + p \nabla \lambda
$

#### 1.2.3 速度更新推导
1. 离散动量方程：`A·U = H - ∇(λp)`
2. 定义中间速度 `HbyA = H/A`。预测步单独求解的显式压力力没有写入保存的动量矩阵。
3. 压力修正后用 `U = HbyA - rAAtU*fvc::grad(lambda*p)` 更新速度，与动量预测使用同一个乘积梯度。

#### 1.2.4 能量方程修正
内能形式：
$
\frac{\partial (\lambda \rho e)}{\partial t} + \nabla \cdot (\lambda \rho e \mathbf{U}) = -\nabla \cdot (\lambda p \mathbf{U}) + \lambda S_e
$

总焓形式：
$
\frac{\partial (\lambda \rho h)}{\partial t} + \nabla \cdot (\lambda \rho h \mathbf{U}) = -\lambda \frac{\partial p}{\partial t} - \mathbf{U} \cdot \nabla (\lambda p) + \lambda S_h
$

## 2. PIMPLE 算法循环过程

### 2.1 主循环流程
1. **时间步进**：`runTime++`
2. **外迭代循环**（nOuterCorrectors）：
   - **动量预测**：求解动量方程得到中间速度
   - **能量预测**：求解能量方程更新热力学状态
   - **压力修正**：
     - 构建压力方程
     - 求解压力方程
     - 更新速度场
     - 修正密度场
   - **内迭代循环**（nCorrectors）：
     - **非正交修正**（nNonOrthogonalCorrectors）
3. **后处理**：更新物理量、写入结果

### 2.2 关键步骤与代码对应

#### 2.2.1 动量预测器（momentumPredictor.C）
```cpp
void ArisaSTALL::momentumPredictor()
{
    volVectorField& U(U_);
    const volScalarField& lambda = lambda_;
    const surfaceScalarField lambdaf(fvc::interpolate(lambda));

    tUEqn = (
        lambda*fvm::ddt(rho, U)       // 时间导数项×lambda
      + fvm::div(lambdaf*phi, U)       // 对流项×面插值lambda
      + lambda*MRF.DDt(rho, U)         // MRF源项×lambda
     ==
        lambda*fvModels().source(rho, U)  // 源项×lambda
    );

    if (pimple.momentumPredictor())
    {
        solve(UEqn == -fvc::grad(lambda*p)); // 压力梯度为-∇(λp)
    }
}
```

#### 2.2.2 能量预测器（thermophysicalPredictor.C）
```cpp
void ArisaSTALL::thermophysicalPredictor()
{
    volScalarField& he = thermo_.he();
    const volScalarField& lambda = lambda_;
    const surfaceScalarField lambdaf(fvc::interpolate(lambda));

    fvScalarMatrix EEqn(
        lambda*fvm::ddt(rho, he)       // 焓时间导数×lambda
      + fvm::div(lambdaf*phi, he)       // 焓对流项×lambdaf
      + lambda*fvc::ddt(rho, K)         // 动能时间导数×lambda
      + fvc::div(lambdaf*phi, K)         // 动能对流项×lambdaf
      + pressureWork                     // 压力功处理
        (
            he.name() == "e"
          ? fvc::div(lambdaf*phi, p/rho)()  // 内能形式
          : (-lambda*dpdt - (U & fvc::grad(lambda*p))).internalField()  // 总焓形式
        )
     ==
        lambda*fvModels().source(rho, he)  // 源项×lambda
    );

    EEqn.solve();
    thermo_.correct();
}
```

#### 2.2.3 压力修正（correctPressure.C）
OpenFOAM v13 通过虚函数 `pressureCorrector()` 进入 PISO 循环。Rotor 37 的无浮力分支在此调用自定义压力修正，首轮密度预测和压力修正后的密度更新都使用开口质量通量。启用浮力时沿用基类的 `p_rgh` 求解路径。

压力未知量仍是热力学压力 `p`。令 `G` 表示面的法向离散梯度，`D` 表示插值到面的 `rho*rAAtU`，`lambda_f` 表示面上的开口比例。每次压力修正用当前迭代的场计算显式余项 `G(lambda*p) - lambda_f*G(p)`。将余项与隐式项合并后，迭代收敛时得到完整的离散乘积梯度。即使初始压力均匀，开口比例的变化也会产生压力通量。

连续方程使用开口质量通量 `lambda_f*phi`。其压力拉普拉斯系数为 `lambda_f*lambda_f*D`，一个因子来自开口面积，另一个来自乘积梯度中的隐式项。最后一次非正交修正将守恒面通量除以 `lambda_f` 后写入 `phi`，供动量、密度和能量方程使用。模型要求流通面满足 `lambda_f > 0`。

## 3. 代码结构与修改点

### 3.1 类定义（ArisaSTALL.H）
```cpp
protected:
    // 堵塞因子场（流体体积分数，0<lambda<=1）
    volScalarField lambda_;
```

### 3.2 构造函数（ArisaSTALL.C）
```cpp
ArisaSTALL::ArisaSTALL(fvMesh& mesh)
:
    fluid(mesh),
    lambda_(
        IOobject(
            "lambda",
            mesh.time().constant(),
            mesh,
            IOobject::READ_IF_PRESENT,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedScalar("lambda", dimless, 1.0)
    )
{
    // lambda不存在时读取fvSolution中的默认值
    if (!lambda_.headerOk())
    {
        IOdictionary dict(IOobject(
            "fvSolution",
            mesh.time().system(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        ));
        scalar lambdaDefault = dict.lookupOrDefault<scalar>("lambdaDefault", 1.0);
        lambda_ = lambdaDefault;
    }
}
```

### 3.3 核心修改点总结
| 文件 | 主要修改 | 代码位置 |
|------|----------|----------|
| momentumPredictor.C | 动量方程添加lambda修正 | 时间导数、对流项、源项、压力梯度 |
| thermophysicalPredictor.C | 能量方程添加lambda修正 | 焓时间导数、对流项、压力功 |
| correctPressure.C | 压力修正与速度更新 | 拉普拉斯系数、压力方程、速度更新 |

## 4. 编译与运行

### 4.1 编译步骤
```bash
# 进入求解器目录
cd d:\src_backup\ArisaSTALL

# 清理旧编译文件
wclean

# 编译
wmake -j

# 验证编译成功
ls $FOAM_USER_LIBBIN | grep libArisaSTALL.so
```

### 4.2 运行示例算例
```bash
# 复制教程算例
cp -r $FOAM_TUTORIALS/compressible/ArisaSTALL/forwardStep ./

# 进入算例目录
cd forwardStep

# 生成网格
blockMesh

# 设置初始条件
cp 0.org 0

# 运行求解
ArisaSTALL  # 串行
# mpirun -np 8 ArisaSTALL -parallel  # 并行
```

## 5. 算例配置

### 5.1 system/fvSolution
```c++
// 全局默认lambda值
lambdaDefault 0.85; // 0~1之间，默认1.0

// PIMPLE配置
PIMPLE
{
    nOuterCorrectors 3;
    nCorrectors      2;
    nNonOrthogonalCorrectors 1;
}
```

### 5.2 constant/lambda
```c++
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "constant";
    object      lambda;
}

dimensions      [0 0 0 0 0 0 0];
internalField   uniform 0.85;

boundaryField
{
    inlet
    {
        type            fixedValue;
        value           uniform 1.0;
    }
    wall
    {
        type            zeroGradient;
    }
}
```
