"use strict";(self.webpackChunkmsamp_website=self.webpackChunkmsamp_website||[]).push([[349],{1922:function(e,t,n){n.r(t),n.d(t,{contentTitle:function(){return u},default:function(){return d},frontMatter:function(){return s},metadata:function(){return l},toc:function(){return c}});var i=n(7462),r=n(3366),o=(n(7294),n(3905)),a=["components"],s={id:"introduction"},u="Introduction",l={unversionedId:"introduction",id:"introduction",isDocsHomePage:!1,title:"Introduction",description:"Features",source:"@site/../docs/introduction.md",sourceDirName:".",slug:"/introduction",permalink:"/MS-AMP/docs/introduction",editUrl:"https://github.com/azure/MS-AMP/edit/main/website/../docs/introduction.md",version:"current",frontMatter:{id:"introduction"},sidebar:"docs",next:{title:"Installation",permalink:"/MS-AMP/docs/getting-started/installation"}},c=[{value:"Features",id:"features",children:[]},{value:"Performance",id:"performance",children:[{value:"Accuracy: no loss of accuracy",id:"accuracy-no-loss-of-accuracy",children:[]},{value:"Memory",id:"memory",children:[]}]}],m={toc:c};function d(e){var t=e.components,s=(0,r.Z)(e,a);return(0,o.kt)("wrapper",(0,i.Z)({},m,s,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("h1",{id:"introduction"},"Introduction"),(0,o.kt)("h2",{id:"features"},"Features"),(0,o.kt)("p",null,(0,o.kt)("strong",{parentName:"p"},"MS-AMP")," is an automatic mixed precision package for deep learning developed by Microsoft:"),(0,o.kt)("p",null,"Features:"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},"Support O1 optimization: Apply FP8 to weights and weight gradients and support FP8 in communication."),(0,o.kt)("li",{parentName:"ul"},"Support O2 optimization: Support FP8 for two optimizers(Adam and AdamW)."),(0,o.kt)("li",{parentName:"ul"},"Support O3 optimization: Support FP8 in DeepSpeed ZeRO optimizer."),(0,o.kt)("li",{parentName:"ul"},"Provide four training examples using FP8: Swin-Transformer, DeiT, RoBERTa and GPT-3.")),(0,o.kt)("p",null,"MS-AMP has the following benefit comparing with Transformer Engine:"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},"Support the new FP8 feature that is introduced by latest accelerators (e.g. H100)."),(0,o.kt)("li",{parentName:"ul"},"Speed up math-intensive operations, such as linear layers, by using Tensor Cores."),(0,o.kt)("li",{parentName:"ul"},"Speed up memory-limited operations by accessing one byte compared to half or single-precision."),(0,o.kt)("li",{parentName:"ul"},"Reduce memory requirements for training models, enabling larger models or larger minibatches."),(0,o.kt)("li",{parentName:"ul"},"Speed up communication for distributed model by transmitting lower precision gradients.")),(0,o.kt)("h2",{id:"performance"},"Performance"),(0,o.kt)("h3",{id:"accuracy-no-loss-of-accuracy"},"Accuracy: no loss of accuracy"),(0,o.kt)("p",null,"We evaluated the training loss and validation performance of three typical models, Swin-Transformer, DeiT and RoBERTa, using both MS-AMP O2 and FP16 AMP. Our observations showed that the models trained with MS-AMP O2 mode achieved comparable performance to those trained using FP16 AMP. This demonstrates the effectiveness of the Mixed FP8 O2 mode in MS-AMP."),(0,o.kt)("p",null,"Here are the results for Swin-T, DeiT-S and RoBERTa-B:"),(0,o.kt)("p",null,(0,o.kt)("img",{alt:"image",src:n(5507).Z})),(0,o.kt)("h3",{id:"memory"},"Memory"),(0,o.kt)("p",null,"MS-AMP preserves 32-bit accuracy while using only a fraction of the memory footprint on a range of tasks, including the DeiT model and Swin Transformer for ImageNet classification. For example, comparing with FP16 AMP, MS-AMP with O2 mode can achieve 44% memory saving for Swin-1.0B and 26% memory saving for ViT-1.2B. The proportion of memory saved will be more obvious for larger models."),(0,o.kt)("p",null,"Here are the results for Swin-1.0B and ViT-1.2B."),(0,o.kt)("p",null,(0,o.kt)("img",{alt:"Image",src:n(5059).Z})),(0,o.kt)("p",null,"For detailed setting and results, please go to ",(0,o.kt)("a",{parentName:"p",href:"https://github.com/Azure/MS-AMP-Examples"},"MS-AMP-Example"),"."))}d.isMDXComponent=!0},5059:function(e,t,n){t.Z=n.p+"assets/images/gpu-memory-4b547791bf85e361ddd5b1f1c3f83892.png"},5507:function(e,t,n){t.Z=n.p+"assets/images/performance-8f0205e5a059a0ceb269eed72b673b0f.png"}}]);