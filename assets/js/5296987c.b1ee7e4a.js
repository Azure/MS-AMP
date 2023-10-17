"use strict";(self.webpackChunkmsamp_website=self.webpackChunkmsamp_website||[]).push([[368],{5183:function(e,t,n){n.r(t),n.d(t,{contentTitle:function(){return i},default:function(){return c},frontMatter:function(){return p},metadata:function(){return d},toc:function(){return r}});var l=n(7462),o=n(3366),a=(n(7294),n(3905)),s=["components"],p={id:"development"},i="Development",d={unversionedId:"developer-guides/development",id:"developer-guides/development",isDocsHomePage:!1,title:"Development",description:"If you want to develop new feature, please follow below steps to set up development environment.",source:"@site/../docs/developer-guides/development.md",sourceDirName:"developer-guides",slug:"/developer-guides/development",permalink:"/MS-AMP/docs/developer-guides/development",editUrl:"https://github.com/azure/MS-AMP/edit/main/website/../docs/developer-guides/development.md",version:"current",frontMatter:{id:"development"},sidebar:"docs",previous:{title:"Container Images",permalink:"/MS-AMP/docs/user-tutorial/container-images"},next:{title:"Using Docker",permalink:"/MS-AMP/docs/developer-guides/using-docker"}},r=[{value:"Check Environment",id:"check-environment",children:[]},{value:"Set up",id:"set-up",children:[]},{value:"Lint and Test",id:"lint-and-test",children:[]}],u={toc:r};function c(e){var t=e.components,n=(0,o.Z)(e,s);return(0,a.kt)("wrapper",(0,l.Z)({},u,n,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("h1",{id:"development"},"Development"),(0,a.kt)("p",null,"If you want to develop new feature, please follow below steps to set up development environment."),(0,a.kt)("p",null,"We suggest you to use ",(0,a.kt)("a",{parentName:"p",href:"https://vscode.github.com/"},"Visual Studio Code")," and install the recommended extensions for this project.\nYou can also develop online with ",(0,a.kt)("a",{parentName:"p",href:"https://github.com/codespaces"},"GitHub Codespaces"),"."),(0,a.kt)("h2",{id:"check-environment"},"Check Environment"),(0,a.kt)("p",null,"Follow ",(0,a.kt)("a",{parentName:"p",href:"/MS-AMP/docs/getting-started/installation"},"System Requirements"),"."),(0,a.kt)("h2",{id:"set-up"},"Set up"),(0,a.kt)("p",null,"Clone code."),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-bash"},"git clone --recurse-submodules https://github.com/azure/MS-AMP\ncd MS-AMP\n")),(0,a.kt)("p",null,"Install MS-AMP."),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-bash"},"python3 -m pip install --upgrade pip\npython3 -m pip install -e .[test] \nmake postinstall\n")),(0,a.kt)("p",null,"Install MSCCL and preload msamp_dist library"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-bash"},'cd third_party/msccl\n# H100\nmake -j src.build NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"\napt-get update\napt install build-essential devscripts debhelper fakeroot\nmake pkg.debian.build\ndpkg -i build/pkg/deb/libnccl2_*.deb\ndpkg -i build/pkg/deb/libnccl-dev_2*.deb\n\ncd -\nNCCL_LIBRARY=/usr/lib/x86_64-linux-gnu/libnccl.so # Change as needed\nexport LD_PRELOAD="/usr/local/lib/libmsamp_dist.so:${NCCL_LIBRARY}:${LD_PRELOAD}"\n')),(0,a.kt)("h2",{id:"lint-and-test"},"Lint and Test"),(0,a.kt)("p",null,"Format code using yapf."),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-bash"},"python3 setup.py format\n")),(0,a.kt)("p",null,"Check code style with mypy and flake8"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-bash"},"python3 setup.py lint\n")),(0,a.kt)("p",null,"Run unit tests."),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-bash"},"python3 setup.py test\n")),(0,a.kt)("p",null,"Open a pull request to main branch on GitHub."))}c.isMDXComponent=!0}}]);