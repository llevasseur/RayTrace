#pragma once

#include <atlas/core/Float.hpp>
#include <atlas/math/Math.hpp>
#include <atlas/math/Random.hpp>
#include <atlas/math/Ray.hpp>

#include <fmt/printf.h>
#include <stb_image.h>
#include <stb_image_write.h>

#include <limits>
#include <memory>
#include <vector>

using atlas::core::areEqual;

using Colour = atlas::math::Vector;

void saveToFile(std::string const& filename,
	std::size_t width,
	std::size_t height,
	std::vector<Colour> const& image);

// Declarations
class BRDF;
class BTDF;
class Camera;
class Material;
class Light;
class Shape;
class Sampler;
class Tracer;

struct World
{
	std::size_t width, height;
	Colour background;
	std::shared_ptr<Sampler> sampler;
	std::vector<std::shared_ptr<Shape>> scene;
	std::vector<Colour> image;
	std::vector<std::shared_ptr<Light>> lights;
	std::shared_ptr<Light> ambient;
	Colour max_to_one(Colour const& c) const;
	std::shared_ptr<Tracer> tracer_ptr;
	int max_depth;
};

class Tracer 
{
public:
	Tracer(std::shared_ptr<World> world_ptr);
	virtual ~Tracer() = default;

	virtual Colour trace_ray(atlas::math::Ray<atlas::math::Vector> const& ray) const;
	virtual Colour trace_ray(atlas::math::Ray<atlas::math::Vector> const& ray, const int depth) const;

protected:
	std::shared_ptr<World> mWorld_ptr;

private:
	Tracer();
};



struct ShadeRec
{
	Colour colour;
	float t;
	atlas::math::Normal normal;
	atlas::math::Ray<atlas::math::Vector> ray;
	std::shared_ptr<Material> material;
	std::shared_ptr<World> world;
	atlas::math::Point hitPoint;
	int depth;
};

// Abstract classes defining the interfaces for concrete entities

class Camera
{
public:
	Camera();

	virtual ~Camera() = default;

	virtual void renderScene(std::shared_ptr<World>& world) const = 0;

	void setEye(atlas::math::Point const& eye);

	void setLookAt(atlas::math::Point const& lookAt);

	void setUpVector(atlas::math::Vector const& up);

	void computeUVW();

protected:
	atlas::math::Point mEye;
	atlas::math::Point mLookAt;
	atlas::math::Point mUp;
	atlas::math::Vector mU, mV, mW;
};

class Sampler
{
public:
	Sampler(int numSamples, int numSets);
	virtual ~Sampler() = default;

	int getNumSamples() const;

	void setupShuffledIndeces();

	virtual void generateSamples() = 0;

	atlas::math::Point sampleUnitSquare();

protected:
	std::vector<atlas::math::Point> mSamples;
	std::vector<int> mShuffledIndeces;

	int mNumSamples;
	int mNumSets;
	unsigned long mCount;
	int mJump;
};

class Shape
{
public:
	Shape();
	virtual ~Shape() = default;

	// if t computed is less than the t in sr, it and the color should be
	// updated in sr
	virtual bool hit(atlas::math::Ray<atlas::math::Vector> const& ray,
		ShadeRec& sr) const = 0;

	virtual bool shadowHit(atlas::math::Ray<atlas::math::Vector> const& shadowRay,
		float& t) const = 0;

	void setColour(Colour const& col);

	Colour getColour() const;

	void setMaterial(std::shared_ptr<Material> const& material);

	std::shared_ptr<Material> getMaterial() const;


protected:
	virtual bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
		float& tMin) const = 0;

	Colour mColour;
	std::shared_ptr<Material> mMaterial;
};

class BRDF
{
public:
	virtual ~BRDF() = default;

	virtual Colour fn(ShadeRec const& sr,
		atlas::math::Vector const& reflected,
		atlas::math::Vector const& incoming) const = 0;
	virtual Colour rho(ShadeRec const& sr,
		atlas::math::Vector const& reflected) const = 0;
};

class BTDF 
{
public:
	virtual ~BTDF() = default;

	virtual Colour fn(ShadeRec const& sr,
		atlas::math::Vector const& reflected,
		atlas::math::Vector const& incoming) const = 0;

	virtual Colour sample_f(ShadeRec const& sr,
		atlas::math::Vector const& reflected,
		atlas::math::Vector& incoming) const = 0;

	virtual Colour rho(ShadeRec const& sr,
		atlas::math::Vector const& reflected) const = 0;

	virtual bool tir(ShadeRec const& sr) const = 0;
};


class Material
{
public:
	virtual ~Material() = default;

	virtual Colour shade(ShadeRec& sr) = 0;
};


class Light
{
public:
	virtual atlas::math::Vector getDirection(ShadeRec& sr) = 0;

	virtual Colour L(ShadeRec& sr);

	void scaleRadiance(float b);

	void setColour(Colour const& c);

	virtual bool inShadow(atlas::math::Ray<atlas::math::Vector> const& ray,
		ShadeRec& sr) const = 0;


protected:
	Colour mColour;
	float mRadiance;
};


// Concrete classes which we can construct and use in our ray tracer

class Whitted : public Tracer
{
public:
	Whitted(std::shared_ptr<World> world_ptr);
	virtual ~Whitted();

	Colour trace_ray(atlas::math::Ray<atlas::math::Vector> const& ray,
		int depth) const override;
private:
	Whitted();
};


class Sphere : public Shape
{
public:
	Sphere(atlas::math::Point center, float radius);

	bool hit(atlas::math::Ray<atlas::math::Vector> const& ray,
		ShadeRec& sr) const;

	bool shadowHit(atlas::math::Ray<atlas::math::Vector> const& shadowRay,
		float& t) const;

private:
	bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
		float& tMin) const;

	atlas::math::Point mCentre;
	float mRadius;
	float mRadiusSqr;
};

class Triangle : public Shape
{
public:
	Triangle(atlas::math::Vector v0, atlas::math::Vector v1, atlas::math::Vector v2);

	bool hit(atlas::math::Ray<atlas::math::Vector> const& ray,
		ShadeRec& sr) const;

	bool shadowHit(atlas::math::Ray<atlas::math::Vector> const& shadowRay,
		float& t) const;

private:
	bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
		float& tMin) const;

	atlas::math::Vector mV0;
	atlas::math::Vector mV1;
	atlas::math::Vector mV2;
};

class Plane : public Shape
{
public:
	Plane(atlas::math::Point p, atlas::math::Vector normal);

	bool hit(atlas::math::Ray<atlas::math::Vector> const& ray,
		ShadeRec& sr) const;

	bool shadowHit(atlas::math::Ray<atlas::math::Vector> const& shadowRay,
		float& t) const;

private:
	bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
		float& tMin) const;

	atlas::math::Point mP;
	atlas::math::Vector mNormal;
};

class Pinhole : public Camera
{
public:
	Pinhole();

	void setDistance(float dist);
	void setZoom(float zoom);

	atlas::math::Vector rayDirection(atlas::math::Point const& p) const;
	void renderScene(std::shared_ptr<World>& world) const;

private:
	float mDist;
	float mZoom;
};

class Regular : public Sampler
{
public:
	Regular(int numSamples, int numSets);

	void generateSamples();
};

class Random : public Sampler
{
public:
	Random(int numSamples, int numSets);

	void generateSamples();
};

class Multijittered : public Sampler
{
public:
	Multijittered(int numSamples, int numSets);

	void generateSamples();

};

class Lambertian : public BRDF
{
public:
	Lambertian();
	Lambertian(Colour diffuseColour, float diffuseReflection);

	Colour fn(ShadeRec const& sr,
		atlas::math::Vector const& reflected,
		atlas::math::Vector const& incoming) const override; //sr, wo, wi

	Colour rho(ShadeRec const& sr,
		atlas::math::Vector const& reflected) const override; //sr, wo

	void setDiffuseReflection(float kd);

	void setDiffuseColour(Colour const& colour);

private:
	Colour mDiffuseColour;
	float mDiffuseReflection;
};

class GlossySpecular : public BRDF
{
public:
	GlossySpecular();
	GlossySpecular(Colour diffuseColour, float exp, float diffuseReflection);

	Colour fn(ShadeRec const& sr,
		atlas::math::Vector const& reflected,
		atlas::math::Vector const& incoming) const override;

	Colour rho(ShadeRec const& sr,
		atlas::math::Vector const& reflected) const override;

	void setExponent(float exp);

	void setDiffuseReflection(float ks);

	void setDiffuseColour(Colour const& colour);

private:
	Colour mDiffuseColour;
	float mDiffuseReflection;
	float mExp;
};


class PerfectSpecular : public BRDF
{
public:
	PerfectSpecular();
	PerfectSpecular(float Kr, Colour Cr);

	Colour fn(ShadeRec const& sr,
		atlas::math::Vector const& reflected,
		atlas::math::Vector const& incoming) const override;

	Colour rho(ShadeRec const& sr,
		atlas::math::Vector const& reflected) const override;

	Colour sample_f(ShadeRec const& sr, 
		atlas::math::Vector const& reflected,
		atlas::math::Vector& incoming) const;

	void set_kr(float kr);

	void set_cr(Colour cr);

private:
	float mKr;
	Colour mCr;

};


class PerfectTransmitter : public BTDF
{
public:
	PerfectTransmitter();
	PerfectTransmitter(float kt, float ior);

	Colour fn(ShadeRec const& sr,
		atlas::math::Vector const& reflected,
		atlas::math::Vector const& incoming) const;

	Colour sample_f(ShadeRec const& sr,
		atlas::math::Vector const& reflected,
		atlas::math::Vector& incoming) const;

	Colour rho(ShadeRec const& sr,
		atlas::math::Vector const& reflected) const;

	bool tir(ShadeRec const& sr) const;

private:
	float mKt;
	float mIor;
};


/*class FresnalTransmitter : public BTDF
{
	FresnalTransmitter();
	FresnalTransmitter(float eta_in, float eta_out);

	Colour fn(ShadeRec const& sr,
		atlas::math::Vector const& reflected,
		atlas::math::Vector const& incoming) const;

	Colour sample_f(ShadeRec const& sr,
		atlas::math::Vector const& reflected,
		atlas::math::Vector& incoming) const;

	Colour rho(ShadeRec const& sr,
		atlas::math::Vector const& reflected) const;

	bool tir(ShadeRec const& sr) const;

private:
	float mEta_in;
	float mEta_out;
};*/



class Matte : public Material
{
public:
	Matte();
	Matte(float kd, float ka, Colour colour);

	void setDiffuseReflection(float k);

	void setAmbientReflection(float k);

	void setDiffuseColour(Colour colour);

	Colour shade(ShadeRec& sr) override;

private:
	std::shared_ptr<Lambertian> mDiffuseBRDF;
	std::shared_ptr<Lambertian> mAmbientBRDF;
};

class Phong : public Material
{
public:
	Phong();
	Phong(float kd, float ka, float exp, Colour colour);

	void setDiffuseReflection(float k);

	void setAmbientReflection(float k);

	void setExponent(float exp);

	void setDiffuseColour(Colour colour);

	Colour shade(ShadeRec& sr) override;

private:
	std::shared_ptr<Lambertian> mDiffuseBRDF;
	std::shared_ptr<Lambertian> mAmbientBRDF;
	std::shared_ptr<GlossySpecular> mSpecularBRDF;
};


class Reflective : public Phong
{
public:
	//Constructors, access functions, etc
	Reflective();
	Reflective(float kd, float ka, float exp, Colour colour, float kr, Colour cr);
	virtual Colour shade(ShadeRec& sr) override;

private:
	std::shared_ptr<PerfectSpecular> mReflectiveBRDF;
};


class Transparent : public Phong
{
public:
	//Constructors, etc.
	Transparent();
	Transparent(float kd, float ka, float exp, 
		Colour colour, float kr, Colour cr, float kt, float ior);
	virtual Colour shade(ShadeRec& sr) override;

private:
	std::shared_ptr<PerfectSpecular> mReflectiveBRDF;
	std::shared_ptr<PerfectTransmitter> mSpecularBTDF;
};




class PointLight : public Light
{
public:
	PointLight();
	PointLight(atlas::math::Point const& origin);

	void setOrigin(atlas::math::Point const& origin);

	atlas::math::Vector getDirection(ShadeRec& sr) override;

	bool inShadow(atlas::math::Ray<atlas::math::Vector> const& ray,
		ShadeRec& sr) const;


private:
	atlas::math::Point getOrigin(ShadeRec& sr);

	atlas::math::Point mOrigin;
};


class Directional : public Light
{
public:
	Directional();
	Directional(atlas::math::Vector const& dir);

	void setDirection(atlas::math::Vector const& dir);

	atlas::math::Vector getDirection(ShadeRec& sr) override;

	bool inShadow(atlas::math::Ray<atlas::math::Vector> const& ray,
		ShadeRec& sr) const;


private:
	atlas::math::Vector mDir;
};


class Ambient : public Light
{
public:
	Ambient();

	atlas::math::Vector getDirection(ShadeRec& sr) override;

	bool inShadow(atlas::math::Ray<atlas::math::Vector> const& ray,
		ShadeRec& sr)const;

};