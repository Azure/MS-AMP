"use strict";(self.webpackChunkmsamp_website=self.webpackChunkmsamp_website||[]).push([[449],{3905:function(e,t,n){n.d(t,{Zo:function(){return s},kt:function(){return d}});var r=n(7294);function a(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function i(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function o(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?i(Object(n),!0).forEach((function(t){a(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):i(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function l(e,t){if(null==e)return{};var n,r,a=function(e,t){if(null==e)return{};var n,r,a={},i=Object.keys(e);for(r=0;r<i.length;r++)n=i[r],t.indexOf(n)>=0||(a[n]=e[n]);return a}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(r=0;r<i.length;r++)n=i[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(a[n]=e[n])}return a}var u=r.createContext({}),c=function(e){var t=r.useContext(u),n=t;return e&&(n="function"==typeof e?e(t):o(o({},t),e)),n},s=function(e){var t=c(e.components);return r.createElement(u.Provider,{value:t},e.children)},m={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},p=r.forwardRef((function(e,t){var n=e.components,a=e.mdxType,i=e.originalType,u=e.parentName,s=l(e,["components","mdxType","originalType","parentName"]),p=c(n),d=a,f=p["".concat(u,".").concat(d)]||p[d]||m[d]||i;return n?r.createElement(f,o(o({ref:t},s),{},{components:n})):r.createElement(f,o({ref:t},s))}));function d(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var i=n.length,o=new Array(i);o[0]=p;var l={};for(var u in t)hasOwnProperty.call(t,u)&&(l[u]=t[u]);l.originalType=e,l.mdxType="string"==typeof e?e:a,o[1]=l;for(var c=2;c<i;c++)o[c]=n[c];return r.createElement.apply(null,o)}return r.createElement.apply(null,n)}p.displayName="MDXCreateElement"},8215:function(e,t,n){var r=n(7294);t.Z=function(e){var t=e.children,n=e.hidden,a=e.className;return r.createElement("div",{role:"tabpanel",hidden:n,className:a},t)}},1395:function(e,t,n){n.d(t,{Z:function(){return s}});var r=n(7294),a=n(944),i=n(6010),o="tabItem_vU9c",l="tabItemActive_cw6a";var u=37,c=39;var s=function(e){var t=e.lazy,n=e.block,s=e.defaultValue,m=e.values,p=e.groupId,d=e.className,f=(0,a.Z)(),v=f.tabGroupChoices,g=f.setTabGroupChoices,b=(0,r.useState)(s),y=b[0],h=b[1],k=r.Children.toArray(e.children),w=[];if(null!=p){var O=v[p];null!=O&&O!==y&&m.some((function(e){return e.value===O}))&&h(O)}var N=function(e){var t=e.currentTarget,n=w.indexOf(t),r=m[n].value;h(r),null!=p&&(g(p,r),setTimeout((function(){var e,n,r,a,i,o,u,c;(e=t.getBoundingClientRect(),n=e.top,r=e.left,a=e.bottom,i=e.right,o=window,u=o.innerHeight,c=o.innerWidth,n>=0&&i<=c&&a<=u&&r>=0)||(t.scrollIntoView({block:"center",behavior:"smooth"}),t.classList.add(l),setTimeout((function(){return t.classList.remove(l)}),2e3))}),150))},C=function(e){var t,n;switch(e.keyCode){case c:var r=w.indexOf(e.target)+1;n=w[r]||w[0];break;case u:var a=w.indexOf(e.target)-1;n=w[a]||w[w.length-1]}null==(t=n)||t.focus()};return r.createElement("div",{className:"tabs-container"},r.createElement("ul",{role:"tablist","aria-orientation":"horizontal",className:(0,i.Z)("tabs",{"tabs--block":n},d)},m.map((function(e){var t=e.value,n=e.label;return r.createElement("li",{role:"tab",tabIndex:y===t?0:-1,"aria-selected":y===t,className:(0,i.Z)("tabs__item",o,{"tabs__item--active":y===t}),key:t,ref:function(e){return w.push(e)},onKeyDown:C,onFocus:N,onClick:N},n)}))),t?(0,r.cloneElement)(k.filter((function(e){return e.props.value===y}))[0],{className:"margin-vert--md"}):r.createElement("div",{className:"margin-vert--md"},k.map((function(e,t){return(0,r.cloneElement)(e,{key:t,hidden:e.props.value!==y})}))))}},9443:function(e,t,n){var r=(0,n(7294).createContext)(void 0);t.Z=r},944:function(e,t,n){var r=n(7294),a=n(9443);t.Z=function(){var e=(0,r.useContext)(a.Z);if(null==e)throw new Error('"useUserPreferencesContext" is used outside of "Layout" component.');return e}},7522:function(e,t,n){n.r(t),n.d(t,{contentTitle:function(){return s},default:function(){return f},frontMatter:function(){return c},metadata:function(){return m},toc:function(){return p}});var r=n(7462),a=n(3366),i=(n(7294),n(3905)),o=n(1395),l=n(8215),u=["components"],c={id:"container-images"},s="Container Images",m={unversionedId:"user-tutorial/container-images",id:"user-tutorial/container-images",isDocsHomePage:!1,title:"Container Images",description:"MS-AMP provides a set of OCI-compliant container images, which are hosted on and GitHub Container Registry.",source:"@site/../docs/user-tutorial/container-images.mdx",sourceDirName:"user-tutorial",slug:"/user-tutorial/container-images",permalink:"/MS-AMP/docs/user-tutorial/container-images",editUrl:"https://github.com/azure/MS-AMP/edit/main/website/../docs/user-tutorial/container-images.mdx",version:"current",frontMatter:{id:"container-images"},sidebar:"docs",previous:{title:"Optimization Level",permalink:"/MS-AMP/docs/user-tutorial/optimization-level"},next:{title:"Development",permalink:"/MS-AMP/docs/developer-guides/development"}},p=[{value:"Stable tagged versions",id:"stable-tagged-versions",children:[]}],d={toc:p};function f(e){var t=e.components,n=(0,a.Z)(e,u);return(0,i.kt)("wrapper",(0,r.Z)({},d,n,{components:t,mdxType:"MDXLayout"}),(0,i.kt)("h1",{id:"container-images"},"Container Images"),(0,i.kt)("p",null,"MS-AMP provides a set of OCI-compliant container images, which are hosted on and ",(0,i.kt)("a",{parentName:"p",href:"https://github.com/azure/MS-AMP/pkgs/container/msamp"},"GitHub Container Registry"),"."),(0,i.kt)("p",null,"You can use MS-AMP image by ",(0,i.kt)("inlineCode",{parentName:"p"},"ghcr.io/azure/msamp:${tag}"),", available tags are listed below for all stable versions."),(0,i.kt)("h2",{id:"stable-tagged-versions"},"Stable tagged versions"),(0,i.kt)(o.Z,{groupId:"gpu-platform",defaultValue:"cuda",values:[{label:"CUDA",value:"cuda"}],mdxType:"Tabs"},(0,i.kt)(l.Z,{value:"cuda",mdxType:"TabItem"},(0,i.kt)("table",null,(0,i.kt)("thead",{parentName:"table"},(0,i.kt)("tr",{parentName:"thead"},(0,i.kt)("th",{parentName:"tr",align:null},"Tag"),(0,i.kt)("th",{parentName:"tr",align:null},"Description"))),(0,i.kt)("tbody",{parentName:"table"},(0,i.kt)("tr",{parentName:"tbody"},(0,i.kt)("td",{parentName:"tr",align:null},"v0.3.0-cuda12.1"),(0,i.kt)("td",{parentName:"tr",align:null},"MS-AMP v0.3.0 with CUDA 12.1")),(0,i.kt)("tr",{parentName:"tbody"},(0,i.kt)("td",{parentName:"tr",align:null},"v0.3.0-cuda11.8"),(0,i.kt)("td",{parentName:"tr",align:null},"MS-AMP v0.3.0 with CUDA 11.8")),(0,i.kt)("tr",{parentName:"tbody"},(0,i.kt)("td",{parentName:"tr",align:null},"v0.2.0-cuda12.1"),(0,i.kt)("td",{parentName:"tr",align:null},"MS-AMP v0.2.0 with CUDA 12.1")),(0,i.kt)("tr",{parentName:"tbody"},(0,i.kt)("td",{parentName:"tr",align:null},"v0.2.0-cuda11.8"),(0,i.kt)("td",{parentName:"tr",align:null},"MS-AMP v0.2.0 with CUDA 11.8")))))))}f.isMDXComponent=!0},6010:function(e,t,n){function r(e){var t,n,a="";if("string"==typeof e||"number"==typeof e)a+=e;else if("object"==typeof e)if(Array.isArray(e))for(t=0;t<e.length;t++)e[t]&&(n=r(e[t]))&&(a&&(a+=" "),a+=n);else for(t in e)e[t]&&(a&&(a+=" "),a+=t);return a}function a(){for(var e,t,n=0,a="";n<arguments.length;)(e=arguments[n++])&&(t=r(e))&&(a&&(a+=" "),a+=t);return a}n.d(t,{Z:function(){return a}})}}]);